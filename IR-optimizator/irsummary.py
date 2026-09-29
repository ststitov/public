#!/usr/bin/env python3
"""
Сводка по импульсной характеристике в PNG.

Панели: АЧХ, разница АЧХ (при сравнении), форма импульса, спад энергии,
энергия по октавным полосам, спектральный распад CSD (для одного файла)
и таблица числовых метрик.

Использование:
    python3 irsummary.py [file1.wav file2.wav ...] [-o out.png]
            [--separate] [--smooth 3] [--tmax 50] [--dpi 150] [--align band|peak]

Без аргументов берутся все *.wav в текущей папке. По умолчанию все файлы
накладываются на одни оси — так сравнивают кабинеты
между собой или исходник с результатом iroptimizator.py. Если в пачке есть
пары «исходник → результат» (X.wav и X_48k24b_984….wav), разница АЧХ
строится для каждого результата относительно его исходника. Флаг --separate
рисует отдельный PNG на каждый файл.

Зависимости: pip install numpy scipy soundfile matplotlib
"""
import argparse
import glob
import os
import re
import sys

import numpy as np
import soundfile as sf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FuncFormatter

from iroptimizator import (_spectrum, detect_band, detect_predelay, octave_smooth,
                           ir_loudness, BAND_THR_DB)

# Категориальная палитра: восемь цветов, различимых и при нарушениях цветового
# зрения; порядок фиксирован — цвет закреплён за позицией файла, а не за рангом
SERIES = ('#2a78d6', '#eb6834', '#1baf7a', '#eda100',
          '#e87ba4', '#008300', '#4a3aa7', '#e34948')
SURFACE = '#fcfcfb'
INK = '#0b0b0b'
INK_MUTED = '#52514e'
GRID = '#dedcd5'
# Последовательная шкала для CSD — один тон, светлый → тёмный
CSD_CMAP = LinearSegmentedColormap.from_list(
    'csd', ['#f4f7fc', '#b9d3f0', '#6ba6e3', '#2a78d6', '#123a6b'])

DEFAULT_SMOOTH = 3        # 1/3 октавы
TMAX_LIMIT_MS = 50.0      # предел авто-подбора окна панелей времени
TMAX_FLOOR_DB = -40.0     # авто-окно ETC: докуда прослеживается спад энергии
IMPULSE_FLOOR_DB = -30.0  # авто-окно панели формы импульса
LABEL_MAX_CHARS = 18      # длиннее — прямая подпись не ставится, хватает легенды
FFT_MIN_BITS = 17         # длина FFT: разрешение по низам, иначе 1/3 октавы ступенчатая
DEFAULT_DPI = 150
FREQ_LO = 20.0
DB_FLOOR = -60.0          # нижняя граница шкал в дБ
CSD_FLOOR_DB = -40.0      # динамический диапазон спектрального распада
CSD_SLICES = 48
TICKS_HZ = (20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000)
# Полуоктавные полосы 31 Гц – 11 кГц: вдвое подробнее октавных. Снизу — с 31 Гц,
# чтобы захватить основной тон нижней струны восьмиструнки (F#1 ≈ 46 Гц, Drop E
# ≈ 41 Гц — полоса 44 Гц, 37–53 Гц); сверху — до 11 кГц, где виден спад динамика.
# Центры — точные 1000·2^(n/2), подписи — округлённые номиналы.
BAND_CENTERS = tuple(1000 * 2 ** (n / 2) for n in range(-10, 8))
BAND_LABELS = ('31', '44', '63', '88', '125', '177', '250', '354', '500', '707',
               '1к', '1.4к', '2к', '2.8к', '4к', '5.7к', '8к', '11к')
BAND_HALF_WIDTH = 2 ** 0.25        # полоса — от fc/2^(1/4) до fc·2^(1/4)
ALIGN_BAND_HZ = (200.0, 2000.0)  # выравнивание кривых АЧХ по среднему уровню здесь
LABEL_GAP_DB = 3.5        # минимальный зазор между прямыми подписями кривых
# Суффикс, который iroptimizator добавляет к имени результата: _48k24b_984…
RESULT_SUFFIX = re.compile(r'_\d+k\d+b_\d+(_.*)?$')


def read_mono(path):
    """Читает WAV и сводит в моно — тем же способом, что iroptimizator."""
    x, sr = sf.read(path)
    return (x.mean(axis=1) if x.ndim > 1 else x).astype(np.float64), sr


def frequency_response(x, sr, fraction):
    """(частоты, уровень в дБ) — АЧХ, нормированная максимумом к 0 дБ."""
    # Длинный zero-pad: на 30 Гц полоса 1/3 октавы уже 7 Гц, и при коротком
    # преобразовании в неё попадает меньше одного бина — кривая идёт ступеньками.
    n = 1 << max(FFT_MIN_BITS, (len(x) - 1).bit_length())
    power = np.abs(np.fft.rfft(x, n)) ** 2
    freqs = np.fft.rfftfreq(n, 1 / sr)
    top = min(sr / 2, 20000.0)
    out = np.logspace(np.log10(FREQ_LO), np.log10(top), 600)
    smoothed = octave_smooth(freqs, power, out, fraction)
    peak = smoothed.max()
    if peak <= 0:
        return out, np.full_like(out, DB_FLOOR)
    return out, 10 * np.log10(smoothed / peak + 1e-30)


def energy_decay(x, sr):
    """(время в мс, огибающая в дБ) — спад энергии импульса.

    Огибающая считается скользящим RMS, а не через преобразование Гильберта:
    оно нелокально и на резком конце файла даёт выброс вверх — несуществующий
    всплеск энергии в последних миллисекундах.
    """
    win = max(1, int(sr * 0.001))               # окно 1 мс
    env = np.sqrt(np.convolve(x ** 2, np.ones(win) / win, mode='valid'))
    t = (np.arange(len(env)) + win / 2) / sr * 1000
    peak = env.max()
    if peak <= 0:
        return t, np.full(len(env), DB_FLOOR)
    return t, 20 * np.log10(env / peak + 1e-30)


def band_energy(x, sr):
    """Энергия по полуоктавным полосам в дБ относительно самой громкой полосы."""
    n = 1 << max(FFT_MIN_BITS, (len(x) - 1).bit_length())
    power = np.abs(np.fft.rfft(x, n)) ** 2
    freqs = np.fft.rfftfreq(n, 1 / sr)
    levels = []
    for fc in BAND_CENTERS:
        m = (freqs >= fc / BAND_HALF_WIDTH) & (freqs < fc * BAND_HALF_WIDTH)
        levels.append(power[m].sum() if m.any() else 0.0)
    levels = np.array(levels)
    top = levels.max()
    if top <= 0:
        return np.full(len(BAND_CENTERS), DB_FLOOR)
    return 10 * np.log10(levels / top + 1e-30)


def spectral_decay(x, sr, tmax_ms, fraction=6):
    """(частоты, времена, матрица дБ) — спектральный распад CSD.

    Для каждого момента берётся остаток импульса от него до конца: так видно,
    какие резонансы продолжают звенеть, когда остальное уже затухло.
    """
    span = min(len(x), int(sr * tmax_ms / 1000))
    starts = np.unique(np.linspace(0, max(span - 1, 1), CSD_SLICES).astype(int))
    top = min(sr / 2, 20000.0)
    out_f = np.logspace(np.log10(max(FREQ_LO, 100.0)), np.log10(top), 300)
    rows, times = [], []
    n = 1 << max(14, (len(x) - 1).bit_length())
    freqs = np.fft.rfftfreq(n, 1 / sr)
    for s in starts:
        seg = x[s:]
        if len(seg) < 16:
            break
        tail = max(8, len(seg) // 8)
        seg = seg.copy()
        seg[-tail:] *= 0.5 * (1 + np.cos(np.linspace(0, np.pi, tail)))
        power = np.abs(np.fft.rfft(seg, n)) ** 2
        rows.append(octave_smooth(freqs, power, out_f, fraction))
        times.append(s / sr * 1000)
    rows = np.array(rows)
    peak = rows.max()
    if peak <= 0:
        return out_f, np.array(times), np.full_like(rows, CSD_FLOOR_DB)
    return out_f, np.array(times), 10 * np.log10(rows / peak + 1e-30)


def metrics(x, sr, fraction):
    """Числовые характеристики IR для таблицы."""
    freqs, db = frequency_response(x, sr, fraction)
    peak_idx = int(np.argmax(db))

    def band_at(level):
        above = np.flatnonzero(db >= level)
        if not above.size:
            return None, None
        return freqs[above[0]], freqs[above[-1]]

    lo3, hi3 = band_at(-3.0)
    lo10, hi10 = band_at(-10.0)

    n = 1 << max(FFT_MIN_BITS, (len(x) - 1).bit_length())
    power = np.abs(np.fft.rfft(x, n)) ** 2
    lin_freqs = np.fft.rfftfreq(n, 1 / sr)
    centroid = (float(np.sum(lin_freqs * power) / np.sum(power))
                if power.sum() > 0 else 0.0)

    # Время спада считается от пика огибающей, а не от начала файла: в тихой
    # предзадержке огибающая около −600 дБ, и поиск с t = 0 сразу находил
    # «спад» (0.5 мс вместо 3.4 при 200 сэмплах тишины).
    t, etc = energy_decay(x, sr)
    top = int(np.argmax(etc)) if len(etc) else 0
    below = np.flatnonzero(etc[top:] <= -20.0)
    t20 = t[top + below[0]] - t[top] if below.size else None

    peak_lin = float(np.max(np.abs(x)))
    predelay = detect_predelay(x)
    return {
        'Длина': f'{len(x)} / {len(x) / sr * 1000:.1f} мс',
        'Частота': f'{sr / 1000:g} кГц',
        'Пик': f'{20 * np.log10(peak_lin):.1f} dBFS' if peak_lin > 0 else '—',
        'Предзад.': (f'{predelay / sr * 1000:.2f} мс' if predelay else 'нет'),
        'Полоса −3 дБ': f'{lo3:.0f}–{hi3:.0f} Гц' if lo3 else '—',
        'Полоса −10 дБ': f'{lo10:.0f}–{hi10:.0f} Гц' if lo10 else '—',
        'Пик АЧХ': f'{freqs[peak_idx]:.0f} Гц',
        'Центроид': f'{centroid:.0f} Гц',
        'Спад −20 дБ': f'{t20:.1f} мс' if t20 is not None else '> длины',
        'Громкость': f'{ir_loudness(x, sr):.1f} dB',
    }


def align_curve(freqs, db, mode):
    """Выравнивание кривой АЧХ для наложения.

    'peak' — максимум к 0 дБ (как было); 'band' — средний уровень в
    ALIGN_BAND_HZ к 0 дБ. При сравнении 'band' честнее: при выравнивании по
    максимуму один резонанс сдвигает всю кривую, и разница тембра
    маскируется разницей пиков.
    """
    if mode == 'band':
        m = (freqs >= ALIGN_BAND_HZ[0]) & (freqs <= ALIGN_BAND_HZ[1])
        if m.any():
            return db - float(np.mean(db[m]))
    return db


def find_pairs(labels):
    """{индекс результата: индекс исходника} для пар «X» → «X_48k24b_984…».

    Результат узнаётся по суффиксу iroptimizator; переименованный результат
    (например, X_48k24b_984_prev) тоже находит свой исходник. При нескольких
    кандидатах берётся самый длинный исходник: «cab_2» точнее, чем «cab».
    """
    pairs = {}
    for i, label in enumerate(labels):
        candidates = [j for j, src in enumerate(labels)
                      if j != i and label.startswith(src + '_')
                      and RESULT_SUFFIX.fullmatch(label[len(src):])]
        if candidates:
            pairs[i] = max(candidates, key=lambda j: len(labels[j]))
    return pairs


def style_axes(ax, title, xlabel, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, fontsize=11, loc='left', pad=8)
    ax.set_xlabel(xlabel, color=INK_MUTED, fontsize=9)
    ax.set_ylabel(ylabel, color=INK_MUTED, fontsize=9)
    ax.tick_params(colors=INK_MUTED, labelsize=8.5, length=3)
    ax.grid(True, color=GRID, linewidth=0.7, alpha=0.9)
    ax.set_axisbelow(True)
    for side, spine in ax.spines.items():
        spine.set_visible(side in ('left', 'bottom'))
        spine.set_color(GRID)


def hz_formatter(v, _):
    return f'{v / 1000:g}к' if v >= 1000 else f'{v:g}'


def auto_tmax(items, floor_db):
    """Окно панели времени: докуда спад энергии держится выше floor_db."""
    spans = []
    for _, x, sr in items:
        t, etc = energy_decay(x, sr)
        above = np.flatnonzero(etc > floor_db)
        spans.append(t[above[-1]] if above.size else (t[-1] if len(t) else 1.0))
    return float(min(TMAX_LIMIT_MS, max(5.0, max(spans) * 1.3)))


def short_label(label, limit=28):
    """Сокращение длинного имени с середины: у результатов iroptimizator
    различается конец (…_984, …_984_prev, …_2048), и обрезка по концу делала
    разные строки таблицы неотличимыми."""
    if len(label) <= limit:
        return label
    head = (limit - 1) // 3
    return label[:head] + '…' + label[-(limit - 1 - head):]


def draw_metrics_table(ax, items, fraction):
    """Таблица метрик. Имя файла — первой колонкой, а не подписью строки:
    подписи строк рисуются вне осей и уезжают за край картинки."""
    ax.axis('off')
    rows = [metrics(x, sr, fraction) for _, x, sr in items]
    keys = list(rows[0])
    table = ax.table(
        cellText=[[short_label(label)] + [r[k] for k in keys]
                  for (label, _, _), r in zip(items, rows)],
        colLabels=['Файл'] + keys, cellLoc='left', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.auto_set_column_width(range(len(keys) + 1))
    table.scale(1, 1.35)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor(GRID)
        cell.set_facecolor(SURFACE)
        cell.get_text().set_color(INK if row > 0 else INK_MUTED)
        if row == 0:
            cell.get_text().set_fontweight('bold')
        elif col == 0:                      # цветной маркер серии у имени файла
            cell.set_edgecolor(SERIES[(row - 1) % len(SERIES)])
            cell.set_linewidth(2.5)


def draw(items, out_path, fraction, imp_ms, etc_ms, dpi, align='band'):
    """items: список (подпись, сигнал, частота). Рисует PNG со сводкой."""
    single = len(items) == 1
    table_share = 0.16 + 0.05 * len(items)
    fig = plt.figure(figsize=(18, 12.5 + 0.35 * len(items)))   # шире: 16 полос × до 8 файлов
    fig.patch.set_facecolor(SURFACE)
    gs = GridSpec(5, 2, figure=fig,
                  height_ratios=[1.15, 0.95, 1.0, 0.9, table_share],
                  hspace=0.6, wspace=0.22)

    ax_fr = fig.add_subplot(gs[0, :])
    ax_second = fig.add_subplot(gs[1, :])          # CSD или разница АЧХ
    ax_imp = fig.add_subplot(gs[2, 0])
    ax_etc = fig.add_subplot(gs[2, 1])
    ax_oct = fig.add_subplot(gs[3, :])
    ax_tab = fig.add_subplot(gs[4, :])

    if single:
        label, x, sr = items[0]
        fig.suptitle(f'{label}   ·   {sr} Гц, {len(x)} сэмпл. '
                     f'({len(x) / sr * 1000:.1f} мс)',
                     color=INK, fontsize=12.5, x=0.01, ha='left')

    responses = [None] * len(items)
    for i, (label, x, sr) in enumerate(items):
        freqs, db = frequency_response(x, sr, fraction)
        responses[i] = (freqs, align_curve(freqs, db, align))
    fr_top = np.ceil(max(db.max() for _, db in responses) + 3)
    fr_floor = fr_top + DB_FLOOR - 3                  # тот же диапазон, что и прежде

    for i, (label, x, sr) in enumerate(items):
        color = SERIES[i % len(SERIES)]
        freqs, db = responses[i]
        ax_fr.semilogx(freqs, np.maximum(db, fr_floor), color=color,
                       linewidth=1.8, label=label)

        t = np.arange(len(x)) / sr * 1000
        keep = t <= imp_ms
        peak = np.max(np.abs(x))
        ax_imp.plot(t[keep], x[keep] / peak if peak > 0 else x[keep],
                    color=color, linewidth=1.4, label=label)

        t_etc, etc = energy_decay(x, sr)
        keep = t_etc <= etc_ms
        ax_etc.plot(t_etc[keep], etc[keep], color=color, linewidth=1.8, label=label)

        # Граница окна: где файл кончается. Для короткого результата обрезки
        # это точка, после которой импульса уже нет.
        end_ms = len(x) / sr * 1000
        for ax, limit in ((ax_imp, imp_ms), (ax_etc, etc_ms)):
            if end_ms < limit:
                ax.axvline(end_ms, color=color, linewidth=1.0, linestyle='--')

    smooth_txt = f'сглаживание 1/{fraction} октавы' if fraction > 0 else 'без сглаживания'
    align_txt = (f'0 дБ — средний уровень {ALIGN_BAND_HZ[0]:.0f} Гц–'
                 f'{ALIGN_BAND_HZ[1] / 1000:g} кГц' if align == 'band' else '0 дБ — максимум')
    style_axes(ax_fr, f'АЧХ ({smooth_txt}; {align_txt})', 'Частота, Гц', 'Уровень, дБ')
    ax_fr.set_xlim(FREQ_LO, max(freqs[-1] for freqs, _ in responses))
    ax_fr.set_ylim(fr_floor, fr_top)
    ax_fr.set_xticks(TICKS_HZ)
    ax_fr.xaxis.set_major_formatter(FuncFormatter(hz_formatter))
    ax_fr.minorticks_off()

    # Ниже удвоенного разрешения окна самого короткого файла форму АЧХ задаёт
    # уже длина окна, а не кабинет (полка и недозвучавший резонанс корпуса).
    shortest = min(items, key=lambda it: len(it[1]) / it[2])
    window_hz = 2 * shortest[2] / len(shortest[1])
    if window_hz > FREQ_LO:
        for ax in (ax_fr,) if single else (ax_fr, ax_second):
            ax.axvspan(FREQ_LO, window_hz, color=GRID, alpha=0.45, linewidth=0, zorder=0)
        ax_fr.text(FREQ_LO * 1.04, fr_top - 1.5,
                   f'ниже {window_hz:.0f} Гц форму задаёт длина окна '
                   f'({len(shortest[1])} сэмпл.)',
                   color=INK_MUTED, fontsize=8, va='top', ha='left')

    if single:
        label, x, sr = items[0]
        # CSD рисуется на окне формы импульса: к этому времени звенеть уже
        # нечему, а растянутая до конца спада панель почти вся пустая.
        f_csd, t_csd, z = spectral_decay(x, sr, imp_ms)
        mesh = ax_second.pcolormesh(f_csd, t_csd, np.maximum(z, CSD_FLOOR_DB),
                                    cmap=CSD_CMAP, vmin=CSD_FLOOR_DB, vmax=0,
                                    shading='auto')
        ax_second.set_xscale('log')
        style_axes(ax_second, 'Спектральный распад (что звенит дольше)',
                   'Частота, Гц', 'Время, мс')
        ax_second.grid(False)
        ax_second.set_xticks([t for t in TICKS_HZ if t >= 100])
        ax_second.xaxis.set_major_formatter(FuncFormatter(hz_formatter))
        ax_second.minorticks_off()
        bar = fig.colorbar(mesh, ax=ax_second, pad=0.01)
        bar.set_label('дБ', color=INK_MUTED, fontsize=9)
        bar.ax.tick_params(colors=INK_MUTED, labelsize=8)
        bar.outline.set_edgecolor(GRID)
    else:
        # Пары «исходник → результат» сравниваются каждая со своим исходником;
        # без пар — все относительно первого файла. За пределами сетки файла
        # (выше его Найквиста при разной частоте дискретизации) разница не
        # рисуется: np.interp подставлял бы последнее значение — выдуманные данные.
        pairs = find_pairs([label for label, _, _ in items])
        comparisons = (sorted(pairs.items()) if pairs
                       else [(i, 0) for i in range(1, len(items))])
        x_hi = FREQ_LO
        for res, src in comparisons:
            base_f, base_db = responses[src]
            freqs, db = responses[res]
            delta = np.interp(base_f, freqs, db, left=np.nan, right=np.nan) - base_db
            ax_second.semilogx(base_f, delta, color=SERIES[res % len(SERIES)],
                               linewidth=1.8, label=f'{items[res][0]} − {items[src][0]}')
            x_hi = max(x_hi, base_f[-1])
        ax_second.axhline(0, color=INK_MUTED, linewidth=1.0)
        title = ('Разница АЧХ: результат − его исходник' if pairs
                 else f'Разница АЧХ относительно «{items[0][0]}»')
        style_axes(ax_second, title, 'Частота, Гц', 'Разница, дБ')
        ax_second.set_xlim(FREQ_LO, x_hi)
        ax_second.set_xticks(TICKS_HZ)
        ax_second.xaxis.set_major_formatter(FuncFormatter(hz_formatter))
        ax_second.minorticks_off()

    has_end = any(len(x) / sr * 1000 < etc_ms for _, x, sr in items)
    style_axes(ax_imp, 'Форма импульса' + (' (пунктир — конец файла)' if has_end else ''),
               'Время, мс', 'Амплитуда')
    ax_imp.set_xlim(0, imp_ms)
    ax_imp.axhline(0, color=GRID, linewidth=0.8)

    style_axes(ax_etc, 'Спад энергии', 'Время, мс', 'Уровень, дБ')
    ax_etc.set_xlim(0, etc_ms)
    ax_etc.set_ylim(DB_FLOOR, 3)

    style_axes(ax_oct, 'Энергия по полуоктавным полосам', 'Полоса, Гц',
               'Уровень, дБ')
    idx = np.arange(len(BAND_CENTERS))
    span = 0.8 / len(items)
    for i, (label, x, sr) in enumerate(items):
        levels = band_energy(x, sr)
        ax_oct.bar(idx - 0.4 + span * (i + 0.5), np.maximum(levels, DB_FLOOR) - DB_FLOOR,
                   bottom=DB_FLOOR, width=span * 0.88, color=SERIES[i % len(SERIES)],
                   label=label, edgecolor=SURFACE, linewidth=1.2)
    # Полосы, центр которых ниже удвоенного разрешения окна самого короткого
    # файла, закрашены, как зона на АЧХ: у короткого IR их уровень задаёт длина
    # окна (полка от незавершённого спада кабинета), а не сам кабинет.
    limited = [k for k, fc in enumerate(BAND_CENTERS) if fc < window_hz]
    if limited:
        ax_oct.axvspan(-0.5, limited[-1] + 0.5, color=GRID, alpha=0.45, linewidth=0, zorder=0)
        ax_oct.text(-0.45, 1.5, f'ниже {window_hz:.0f} Гц уровень задаёт длина окна '
                    f'({len(shortest[1])} сэмпл.)', color=INK_MUTED, fontsize=8,
                    va='top', ha='left')
    ax_oct.set_xticks(idx)
    ax_oct.set_xticklabels(BAND_LABELS)
    ax_oct.set_xlim(-0.5, len(BAND_CENTERS) - 0.5)
    ax_oct.set_ylim(DB_FLOOR, 3)
    ax_oct.grid(True, axis='y', color=GRID, linewidth=0.7)
    ax_oct.grid(False, axis='x')

    ax_tab.set_title('Метрики', color=INK, fontsize=11, loc='left', pad=26)
    draw_metrics_table(ax_tab, items, fraction)

    if len(items) > 1:
        # Легенды — туда, где кривые ровные: расхождения обычно на низах
        # (слева), а внизу по центру АЧХ при выравнивании по полосе пусто.
        ax_fr.legend(loc='lower center', frameon=True, facecolor=SURFACE,
                     edgecolor=GRID, fontsize=9, labelcolor=INK)
        ax_second.legend(loc='upper right', frameon=True, facecolor=SURFACE,
                         edgecolor=GRID, fontsize=8.5, labelcolor=INK)
        # Прямые подписи — только когда имена короткие: длинные налезают на
        # кривые, а идентичность уже несёт легенда. Подпись ставится внутри
        # оси, чуть левее правого края: у самого края с ha='left' она целиком
        # оказывалась за осью и обрезалась — подписи не было видно никогда.
        if len(items) <= 4 and all(len(l) <= LABEL_MAX_CHARS for l, _, _ in items):
            lo, hi = ax_fr.get_ylim()
            x_lab = ax_fr.get_xlim()[1] / 1.12
            marks = sorted((min(max(float(np.interp(x_lab, freqs, db)), lo + 2), hi - 2), label)
                           for (label, _, _), (freqs, db) in zip(items, responses))
            ys = [y for y, _ in marks]
            for k in range(1, len(ys)):          # раздвинуть, чтобы не наезжали
                ys[k] = max(ys[k], ys[k - 1] + LABEL_GAP_DB)
            shift = max(0.0, ys[-1] - (hi - 2))
            for y, (_, label) in zip(ys, marks):
                ax_fr.text(x_lab, y - shift, label, color=INK, fontsize=8.5,
                           va='center', ha='right', clip_on=True,
                           bbox=dict(facecolor=SURFACE, edgecolor='none', alpha=0.85, pad=1.2))

    # tight_layout здесь неприменим (таблица и colorbar с ним несовместимы),
    # поэтому поля задаются явно, а лишние края снимает bbox_inches='tight'.
    fig.subplots_adjust(left=0.07, right=0.985, top=0.96 if single else 0.985,
                        bottom=0.035)
    fig.savefig(out_path, dpi=dpi, facecolor=SURFACE, bbox_inches='tight')
    plt.close(fig)


def describe(x, sr):
    """Строки отчёта по файлу — те же метрики, что печатает iroptimizator."""
    freqs, db = _spectrum(x, sr, window=False)
    band = detect_band(freqs, db)
    predelay = detect_predelay(x)
    lines = [f'{len(x)} сэмпл. ({len(x) / sr * 1000:.1f} мс), '
             f'полоса до {band / 1000:.1f} кГц (порог {BAND_THR_DB:g} dB)']
    if predelay:
        lines.append(f'предзадержка {predelay} сэмпл. '
                     f'({predelay / sr * 1000:.2f} мс)')
    return lines


def main():
    ap = argparse.ArgumentParser(
        description='Сводка по IR в PNG: АЧХ, распад, импульс, октавы, метрики.')
    ap.add_argument('files', nargs='*',
                    help='WAV-файлы; без аргументов — все *.wav в текущей папке')
    ap.add_argument('-o', '--out', default=None,
                    help='PNG (при наложении) или каталог (при --separate)')
    ap.add_argument('--separate', action='store_true',
                    help='отдельный PNG на каждый файл вместо наложения')
    ap.add_argument('--smooth', type=int, default=DEFAULT_SMOOTH,
                    help=f'сглаживание 1/N октавы, 0 — без сглаживания '
                         f'(по умолчанию {DEFAULT_SMOOTH})')
    ap.add_argument('--tmax', type=float, default=None,
                    help='окно панелей времени, мс (по умолчанию подбирается '
                         f'по спаду энергии, не больше {TMAX_LIMIT_MS:g})')
    ap.add_argument('--dpi', type=int, default=DEFAULT_DPI,
                    help=f'разрешение PNG (по умолчанию {DEFAULT_DPI})')
    ap.add_argument('--align', choices=('band', 'peak'), default='band',
                    help=f'выравнивание кривых АЧХ: band — средний уровень '
                         f'{ALIGN_BAND_HZ[0]:.0f} Гц–{ALIGN_BAND_HZ[1] / 1000:g} кГц к 0 дБ '
                         f'(по умолчанию), peak — максимум к 0 дБ')
    a = ap.parse_args()

    auto = not a.files
    if auto:
        a.files = sorted(glob.glob('*.wav'))
        if not a.files:
            sys.exit('Ошибка: файлы не указаны, а в текущей папке нет ни одного *.wav.')
        print(f'Файлы не указаны — беру все *.wav в текущей папке: {len(a.files)} шт.\n')
    missing = [f for f in a.files if not os.path.isfile(f)]
    if missing:
        sys.exit('Файлы не найдены: ' + ', '.join(missing))
    if a.smooth < 0:
        sys.exit('Ошибка: --smooth не может быть отрицательным.')
    if a.tmax is not None and a.tmax <= 0:
        sys.exit('Ошибка: --tmax должен быть больше нуля.')
    if a.separate:
        # Имя картинки строится из имени файла без каталога: a/cab.wav и
        # b/cab.wav затёрли бы друг друга — останавливаемся до отрисовки.
        stems = {}
        for f in a.files:
            stems.setdefault(os.path.splitext(os.path.basename(f))[0], []).append(f)
        clashes = {stem: fs for stem, fs in stems.items() if len(fs) > 1}
        if clashes:
            sys.exit('Ошибка: у разных файлов совпадёт имя картинки, вторая затёрла бы '
                     'первую:\n' + '\n'.join(f'  {stem}: ' + ', '.join(fs)
                                              for stem, fs in clashes.items()))
    if not a.separate and len(a.files) > len(SERIES):
        source = 'в текущей папке *.wav' if auto else 'передано'
        sys.exit(f'Ошибка: на одном графике различимо не больше {len(SERIES)} '
                 f'кривых, а {source} {len(a.files)}. Используйте --separate '
                 f'(отдельная картинка на каждый файл) или укажите файлы явно.')

    items = []
    for i, f in enumerate(a.files, 1):
        x, sr = read_mono(f)
        label = os.path.splitext(os.path.basename(f))[0]
        header = label if len(a.files) == 1 else f'[{i}/{len(a.files)}] {label}'
        print(f'{header}  ({sr} Гц)')
        for line in describe(x, sr):
            print(f'  {line}')
        items.append((label, x, sr))

    if a.separate:
        outdir = a.out or '.'
        os.makedirs(outdir, exist_ok=True)
        for item in items:
            out = os.path.join(outdir, f'{item[0]}_summary.png')
            draw([item], out, a.smooth,
                 a.tmax or auto_tmax([item], IMPULSE_FLOOR_DB),
                 a.tmax or auto_tmax([item], TMAX_FLOOR_DB), a.dpi, a.align)
            print(f'\n→ {out}')
    else:
        out = a.out or 'irsummary.png'
        pairs = find_pairs([label for label, _, _ in items])
        if pairs:
            print('\nПары «исходник → результат»: '
                  + ', '.join(f'{items[s][0]} → {items[r][0]}' for r, s in sorted(pairs.items())))
        draw(items, out, a.smooth,
             a.tmax or auto_tmax(items, IMPULSE_FLOOR_DB),
             a.tmax or auto_tmax(items, TMAX_FLOOR_DB), a.dpi, a.align)
        print(f'\n→ {out}')


if __name__ == '__main__':
    main()
