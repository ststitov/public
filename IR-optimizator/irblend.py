#!/usr/bin/env python3
"""
Объединение двух IR в один — двумя способами.

Параллельно (--mode parallel, по умолчанию) — бленд двух микрофонов или
кабинетов: импульсы складываются. Оба приводятся к общей частоте,
выравниваются по времени (срез предзадержки + подстройка по взаимной
корреляции), проверяются по фазе (противофазный B переворачивается,
гребёнка от сложения отмечается), выравниваются по громкости и смешиваются
в заданной пропорции.

Последовательно (--mode serial) — цепочка «усилитель → кабинет»: сигнал
проходит через оба узла по очереди, и общий IR — это свёртка A ∗ B.
Выравнивание и баланс здесь не нужны: суммирования нет.

Затем — обрезка, fade-out и нормализация, как в iroptimizator.py.
Параметры запрашиваются интерактивно; Enter принимает значение по
умолчанию. Любой параметр можно задать флагом — тогда вопрос не задаётся.

Использование:
    python3 irblend.py A.wav B.wav [-o outdir] [--mode parallel|serial]
            [--sr 44|48] [--bits 16|24] [--len 512|984|1024|2048]
            [--mix 40/60] [--norm peak|loudness] [--loudness-target 8]
            [--fix-polarity] [--no-dither]

Зависимости: pip install numpy scipy soundfile soxr
"""
import argparse
import os
import re
import sys

import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve

from iroptimizator import (
    BITS_CHOICES, DEFAULT_BITS, DEFAULT_LEN, DEFAULT_LOUDNESS_TARGET,
    DEFAULT_NORM, DEFAULT_SR, LEN_CHOICES, LOW_CORR, NORM_CHOICES, PEAK_DBFS,
    SR_CHOICES, SUBTYPES, _batch_default, ask_choice,
    detect_channel_conflict, detect_predelay, ir_loudness, octave_smooth,
    parse_bits, parse_len, parse_sr, process, quantize_pcm16, resample,
    resampler_name,
)

MODE_CHOICES = ('parallel', 'serial')
DEFAULT_MODE = 'parallel'
DEFAULT_MIX = 50.0            # доля A, %
# Кабинет с близким микрофоном к 8 кГц спадает на десятки дБ (динамик),
# а IR усилителя — нет: у него пологая «улыбка» на краях. По этому видно,
# не перепутан ли режим.
CAB_HF_HZ = 8000.0
CAB_HF_DROP_DB = -12.0        # спад на CAB_HF_HZ ниже этого — похоже на кабинет
TONE_BAND_HZ = (40.0, 12000.0)
TONE_REF_HZ = (200.0, 2000.0)  # опорная полоса окраски: её средний уровень — 0 дБ
ALIGN_WINDOW_MS = 5.0         # по какому началу импульса подстраивается сдвиг
ALIGN_RANGE_MS = 2.0          # в каких пределах ищется сдвиг после среза предзадержки
NOTCH_WARN_DB = -6.0          # провал от сложения глубже этого — гребёнка
NOTCH_BAND_HZ = (100.0, 10000.0)
# Суффикс, который iroptimizator добавляет к имени результата: _48k24b_984…
RESULT_SUFFIX = re.compile(r'_\d+k\d+b_\d+(_.*)?$')


# ---------------------------------------------------------------- интерактив

def mix_from_text(raw):
    """Баланс из текста: доля A в процентах. ValueError с пояснением — если ввод неверен.

    Принимаются «40» (доля A, %), «40/60» и «40:60» (пропорция A/B). Сумма
    в пропорции не обязана быть 100: «2/1» — это 66.7/33.3. Результат
    округляется до десятой: он же идёт в имя файла.
    """
    text = raw.strip().replace(',', '.').replace('%', '').replace(' ', '')
    try:
        parts = [float(v) for v in re.split(r'[/:]', text)]
    except ValueError:
        raise ValueError('ожидается доля A («40») или пропорция A/B («40/60»)') from None
    if len(parts) == 1:
        share = parts[0]
    elif len(parts) == 2:
        a, b = parts
        if a <= 0 or b <= 0:
            raise ValueError('обе части пропорции должны быть больше 0')
        share = a / (a + b) * 100
    else:
        raise ValueError('ожидается доля A («40») или пропорция A/B («40/60»)')
    if not 0 < share < 100:
        raise ValueError('доля A должна быть больше 0 и меньше 100 — иначе смешивать нечего')
    return round(share, 1)


def ask_mix():
    """Баланс интерактивно. Enter — 50/50."""
    if not sys.stdin.isatty():
        _batch_default('Баланс', f'{DEFAULT_MIX:g}/{100 - DEFAULT_MIX:g}')
        return DEFAULT_MIX
    while True:
        raw = input(f'\nБаланс A/B — пропорция «40/60» или доля A в процентах «40» '
                    f'[Enter = {DEFAULT_MIX:g}/{100 - DEFAULT_MIX:g}]: ').strip()
        if not raw:
            return DEFAULT_MIX
        try:
            return mix_from_text(raw)
        except ValueError as e:
            print(f'Некорректный ввод: {e}. Повторите.')


def parse_mix(value):
    try:
        return mix_from_text(value)
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e))


# ---------------------------------------------------------------- имя файла

def cab_name(label):
    """Имя кабинета из имени файла: без суффикса прежней конвертации."""
    return RESULT_SUFFIX.sub('', label)


def blend_name(a, b, joiner='+'):
    """«Префикс + A + различающаяся часть B»: общий префикс пишется один раз.

    Префикс обрезается по последнему разделителю, чтобы не резать слово:
    BD_DT_21GetDown + BD_HV_3_AllYouNeed → BD_DT_21GetDown+HV_3_AllYouNeed.
    joiner — «+» для сложения (параллельно), «_x_» для свёртки (последовательно).
    """
    a, b = cab_name(a), cab_name(b)
    prefix = os.path.commonprefix([a, b])
    cut = max(prefix.rfind(sep) for sep in '_- ') + 1
    tail = b[cut:] or b
    return f'{a}{joiner}{tail}'


# ------------------------------------------------------------ окраска АЧХ

def tone_profile(x, sr):
    """(сетка частот, уровень дБ) — 1/3-октавная АЧХ, 0 дБ = средний в TONE_REF_HZ."""
    size = 1 << max(17, (len(x) - 1).bit_length())
    freqs = np.fft.rfftfreq(size, 1 / sr)
    grid = np.logspace(np.log10(TONE_BAND_HZ[0]),
                       np.log10(min(TONE_BAND_HZ[1], sr / 2 * 0.9)), 240)
    power = octave_smooth(freqs, np.abs(np.fft.rfft(x, size)) ** 2, grid, 3)
    db = 10 * np.log10(power + 1e-30)
    ref = (grid >= TONE_REF_HZ[0]) & (grid <= TONE_REF_HZ[1])
    return grid, db - float(np.mean(db[ref]))


def looks_like_cab(x, sr):
    """Спадает ли IR к CAB_HF_HZ так, как спадает динамик в кабинете."""
    grid, db = tone_profile(x, sr)
    return float(np.interp(CAB_HF_HZ, grid, db)) < CAB_HF_DROP_DB


def tone_summary(x, sr):
    """Строка: самый большой подъём и спад АЧХ относительно средней полосы."""
    grid, db = tone_profile(x, sr)
    up, down = int(np.argmax(db)), int(np.argmin(db))
    return (f'окраска АЧХ (0 dB = средний уровень {TONE_REF_HZ[0]:.0f} Гц–'
            f'{TONE_REF_HZ[1] / 1000:g} кГц): подъём до {db[up]:+.1f} dB на {grid[up]:.0f} Гц, '
            f'спад до {db[down]:+.1f} dB на {grid[down]:.0f} Гц')


# ------------------------------------------------------------ выравнивание

def align_pair(a, b, sr):
    """Выравнивание B по A. Возвращает (a, b, отчёт).

    Шаг 1 — срез предзадержки у обоих (порог PREDELAY_THR_DB от пика):
    начала импульсов совпадают. Шаг 2 — подстройка по взаимной корреляции
    первых ALIGN_WINDOW_MS в пределах ±ALIGN_RANGE_MS: порог может сработать
    на тихом предвестнике, а слышно совпадение основной энергии. Знак
    корреляции в найденной точке — фаза: при минусе B переворачивается.
    """
    pa, pb = detect_predelay(a), detect_predelay(b)
    a, b = a[pa:], b[pb:]
    w = max(8, int(sr * ALIGN_WINDOW_MS / 1000))
    lag_max = max(1, int(sr * ALIGN_RANGE_MS / 1000))
    aw = a[:w]
    padded = np.concatenate([np.zeros(lag_max), b[:w + lag_max]])
    padded = np.pad(padded, (0, max(0, len(aw) + 2 * lag_max - len(padded))))
    # c[k] = Σ aw[n]·b[n + k − lag_max] — корреляция при сдвиге B на k − lag_max
    c = np.correlate(padded, aw, mode='valid')[:2 * lag_max + 1]
    best = int(np.argmax(np.abs(c)))
    lag = best - lag_max
    if lag > 0:                          # совпадающая энергия B позже — сдвинуть раньше
        b = b[lag:]
    elif lag < 0:
        b = np.concatenate([np.zeros(-lag), b])
    seg_b = b[:len(aw)]
    denom = np.sqrt(np.sum(aw ** 2) * np.sum(seg_b ** 2))
    corr = float(np.sum(aw * seg_b) / denom) if denom > 0 else 0.0
    flipped = corr < 0
    if flipped:
        b = -b
    return a, b, {'predelay_a': pa, 'predelay_b': pb, 'lag': lag,
                  'corr': corr, 'flipped': flipped}


def notch_check(a, b, sr, n):
    """Самый глубокий провал от сложения в NOTCH_BAND_HZ: (дБ, частота).

    Сравнивается мощность смеси с суммой мощностей слагаемых (1/6 октавы):
    где фазы расходятся, смесь тише суммы — это гребёнка. Где совпадают —
    громче, до +3 дБ при равных уровнях.
    """
    size = 1 << max(17, (n - 1).bit_length())
    freqs = np.fft.rfftfreq(size, 1 / sr)
    hi = min(NOTCH_BAND_HZ[1], sr / 2 * 0.9)
    grid = np.logspace(np.log10(NOTCH_BAND_HZ[0]), np.log10(hi), 300)
    spec = lambda s: octave_smooth(freqs, np.abs(np.fft.rfft(s[:n], size)) ** 2, grid, 6)
    ratio = 10 * np.log10((spec(a + b) + 1e-30) / (spec(a) + spec(b) + 1e-30))
    i = int(np.argmin(ratio))
    return float(ratio[i]), float(grid[i])


# ----------------------------------------------------------------------- CLI

def load(path, target_sr, say):
    """Чтение, свод в моно и ресемплинг одного входа; строки отчёта — в say."""
    x, sr = sf.read(path)
    conflict = detect_channel_conflict(x)
    if conflict is not None:
        corr = conflict[0]
        if corr < LOW_CORR:
            say(f'каналы слабо коррелированы или в противофазе (корреляция '
                f'{corr:.2f}) — свод в моно портит IR ещё до смешивания', warn=True)
        else:
            say(f'{x.shape[1]} канала сведены в моно (корреляция {corr:.2f})')
    mono = (x.mean(axis=1) if x.ndim > 1 else x).astype(np.float64)
    frames = len(mono)
    if sr != target_sr:
        say(f'ресемплинг {sr} → {target_sr} Гц ({resampler_name()})')
        mono = resample(mono, sr, target_sr)
    peak = np.max(np.abs(mono))
    if peak <= 0:
        sys.exit(f'Ошибка: в {path} нет сигнала — смешивать нечего.')
    return mono, sr, frames


def main():
    ap = argparse.ArgumentParser(
        description='Объединение двух IR: параллельно (бленд кабинетов с выравниванием '
                    'и проверкой фазы) или последовательно (свёртка «усилитель → кабинет»).')
    ap.add_argument('files', nargs='+', help='ровно два WAV-файла: A и B')
    ap.add_argument('-o', '--outdir', default='.', help='каталог вывода')
    ap.add_argument('--mode', choices=MODE_CHOICES,
                    help='parallel — сложение двух кабинетов/микрофонов (по умолчанию); '
                         'serial — свёртка «усилитель → кабинет»')
    ap.add_argument('--sr', type=parse_sr, help=f'частота: 44 или 48 (кГц), '
                                                f'по умолчанию {DEFAULT_SR // 1000}')
    ap.add_argument('--bits', type=parse_bits,
                    help=f'битность: 16 или 24, по умолчанию {DEFAULT_BITS}')
    ap.add_argument('--len', dest='length', type=parse_len,
                    help=f'длина результата в сэмплах, по умолчанию {DEFAULT_LEN}')
    ap.add_argument('--mix', type=parse_mix,
                    help=f'баланс: пропорция A/B («40/60», «40:60») или доля A в процентах '
                         f'(«40»), по умолчанию {DEFAULT_MIX:g}/{100 - DEFAULT_MIX:g}')
    ap.add_argument('--norm', choices=NORM_CHOICES,
                    help='нормализация результата: peak (по умолчанию) или loudness')
    ap.add_argument('--loudness-target', type=float,
                    help=f'цель для --norm loudness, дБ (по умолчанию '
                         f'{DEFAULT_LOUDNESS_TARGET:g}); без --norm включает loudness')
    ap.add_argument('--fix-polarity', action='store_true',
                    help='перевернуть результат, если главный пик отрицательный')
    ap.add_argument('--no-dither', action='store_true',
                    help='не применять TPDF-дизеринг при 16 битах')
    a = ap.parse_args()

    if len(a.files) != 2:
        sys.exit(f'Ошибка: нужно ровно два файла — A и B, передано {len(a.files)}.')
    missing = [f for f in a.files if not os.path.isfile(f)]
    if missing:
        sys.exit('Файлы не найдены: ' + ', '.join(missing))
    if a.mix is not None and a.mode == 'serial':
        sys.exit('Ошибка: --mix задаёт баланс сложения, а при --mode serial импульсы '
                 'сворачиваются — смешивать нечего.')
    if a.loudness_target is not None and a.norm == 'peak':
        sys.exit('Ошибка: --loudness-target задаёт цель нормализации по громкости, '
                 'а выбрано --norm peak.')
    if a.loudness_target is not None and a.norm is None:
        a.norm = 'loudness'
    loudness_target = (a.loudness_target if a.loudness_target is not None
                       else DEFAULT_LOUDNESS_TARGET)

    # --mix без --mode — это однозначно бленд: вопрос о режиме не нужен
    mode = a.mode or ('parallel' if a.mix is not None else ask_choice(
        'Режим', MODE_CHOICES, DEFAULT_MODE,
        ['параллельно — два кабинета или микрофона (сложение)',
         'последовательно — усилитель → кабинет (свёртка)']))
    serial = mode == 'serial'
    target_sr = a.sr or ask_choice(
        'Частота дискретизации', SR_CHOICES, DEFAULT_SR,
        [f'{sr // 1000}{"." + str(sr % 1000 // 100) if sr % 1000 else ""} кГц'
         for sr in SR_CHOICES])
    bits = a.bits or ask_choice('Битность', BITS_CHOICES, DEFAULT_BITS,
                                [f'{b} бит' for b in BITS_CHOICES])
    length = a.length or ask_choice(
        'Длина результата', LEN_CHOICES, DEFAULT_LEN,
        [f'{n} сэмпл. ({n / target_sr * 1000:.1f} мс @ {target_sr} Гц)'
         for n in LEN_CHOICES])
    mix = None if serial else (a.mix if a.mix is not None else ask_mix())
    norm = a.norm or ask_choice(
        'Нормализация', NORM_CHOICES, DEFAULT_NORM,
        [f'по пику ({PEAK_DBFS:g} dBFS)',
         f'по громкости (цель {loudness_target:g} dB, пик не выше {PEAK_DBFS:g} dBFS)'])
    dither = bits == 16 and not a.no_dither

    print(f'\nПараметры: {target_sr} Гц, {bits} бит, {length} сэмпл. '
          f'({length / target_sr * 1000:.1f} мс), '
          + ('последовательно (свёртка), ' if serial
             else f'параллельно, баланс {mix:g}/{100 - mix:g}, ')
          + f'нормализация {f"по пику {PEAK_DBFS:g} dBFS" if norm == "peak" else f"по громкости {loudness_target:g} dB"}'
          + (', исправление полярности' if a.fix_polarity else '')
          + (', TPDF-дизеринг' if dither else '') + '\n')

    warned = False

    def block(title):
        lines = []

        def say(text, warn=False):
            nonlocal warned
            lines.append(f'ВНИМАНИЕ — {text}' if warn else text)
            warned = warned or warn
        return lines, say

    # --- входы
    signals, cab_like = [], []
    for tag, path in zip('AB', a.files):
        lines, say = block(tag)
        mono, sr, frames = load(path, target_sr, say)
        predelay = detect_predelay(mono)
        if predelay:
            say(f'предзадержка {predelay} сэмпл. ({predelay / target_sr * 1000:.2f} мс) '
                f'— будет срезана')
        if mono[int(np.argmax(np.abs(mono)))] < 0:
            say('главный пик отрицательный — полярность инвертирована')
        say(f'громкость {ir_loudness(mono, target_sr):.1f} dB (K-взвеш. энергия)')
        say(tone_summary(mono, target_sr))
        cab_like.append(looks_like_cab(mono, target_sr))
        print(f'{tag}: {os.path.basename(path)}  ({sr} Гц, {frames} сэмпл.)')
        for line in lines:
            print(f'  {line}')
        print()
        signals.append(mono)

    # --- не перепутан ли режим: у кабинета к 8 кГц глубокий спад, у усилителя нет
    lines, say = block('режим')
    names = [os.path.basename(f) for f in a.files]
    if serial and all(cab_like):
        say('оба файла похожи на кабинеты (к 8 кГц спад глубже '
            f'{-CAB_HF_DROP_DB:g} dB) — последовательная свёртка двух кабинетов '
            'даёт узкий «коробочный» звук. Для бленда кабинетов нужен параллельный '
            'режим.', warn=True)
    elif not serial and cab_like.count(True) == 1:
        amp = names[cab_like.index(False)]
        say(f'{amp} не похож на кабинет (к 8 кГц нет спада динамика) — если это '
            f'IR усилителя, сложение с кабинетом не имеет смысла: сигнал идёт через '
            f'усилитель и кабинет по очереди, нужен последовательный режим '
            f'(--mode serial).', warn=True)
    if lines:
        print('Проверка режима')
        for line in lines:
            print(f'  {line}')
        print()

    if serial:
        # --- свёртка: предзадержки складываются, поэтому срезаются у обоих
        lines, say = block('свёртка')
        p_a, p_b = detect_predelay(signals[0]), detect_predelay(signals[1])
        x_a, x_b = signals[0][p_a:], signals[1][p_b:]
        say(f'предзадержка срезана: A {p_a} сэмпл., B {p_b} сэмпл.')
        blend = fftconvolve(x_a, x_b)
        say(f'свёртка A ∗ B: {len(x_a)} + {len(x_b)} − 1 = {len(blend)} сэмпл. '
            f'({len(blend) / target_sr * 1000:.1f} мс)')
        say(tone_summary(blend, target_sr).replace('окраска АЧХ', 'итоговая окраска АЧХ'))
        print('Свёртка')
        for line in lines:
            print(f'  {line}')
        print()
    else:
        # --- выравнивание, фаза, громкость, смешивание
        lines, say = block('выравнивание')
        x_a, x_b, info = align_pair(signals[0], signals[1], target_sr)
        to_ms = lambda n: n / target_sr * 1000
        say(f'предзадержка срезана: A {info["predelay_a"]} сэмпл., B {info["predelay_b"]} сэмпл.')
        lag = info['lag']
        say('подстройка по корреляции первых {:g} мс: '.format(ALIGN_WINDOW_MS)
            + (f'B сдвинут на {lag:+d} сэмпл. ({to_ms(lag):+.3f} мс)' if lag
               else 'сдвиг не нужен')
            + f', корреляция {info["corr"]:+.2f}')
        if info['flipped']:
            say('B в противофазе с A — перевёрнут, иначе при смешивании импульсы '
                'гасили бы друг друга')

        # Громкость выравнивается до баланса: 50/50 — поровну на слух, а не по пику
        l_a, l_b = ir_loudness(x_a, target_sr), ir_loudness(x_b, target_sr)
        x_a = x_a / 10 ** (l_a / 20)
        x_b = x_b / 10 ** (l_b / 20)
        say(f'громкости выровнены: B {l_b - l_a:+.1f} dB относительно A до выравнивания')
        w_a, w_b = mix / 100, 1 - mix / 100
        n = max(len(x_a), len(x_b))
        part_a = np.pad(w_a * x_a, (0, n - len(x_a)))
        part_b = np.pad(w_b * x_b, (0, n - len(x_b)))

        window = min(length, n)
        seg_a, seg_b = part_a[:window], part_b[:window]
        denom = np.sqrt(np.sum(seg_a ** 2) * np.sum(seg_b ** 2))
        corr_window = float(np.sum(seg_a * seg_b) / denom) if denom > 0 else 0.0
        dip, dip_hz = notch_check(part_a, part_b, target_sr, window)
        dip_txt = (f'самый глубокий провал от сложения {dip:+.1f} dB на {dip_hz:.0f} Гц'
                   if dip < -1.0 else 'заметных провалов от сложения нет')
        say(f'корреляция A и B в окне {length} сэмпл.: {corr_window:+.2f}; {dip_txt}')
        if dip < NOTCH_WARN_DB:
            say(f'гребёнка: на {dip_hz:.0f} Гц смесь тише суммы слагаемых на {-dip:.1f} dB — '
                f'фазы A и B там расходятся (разные микрофоны или их положение). '
                f'Это может быть задуманным звуком бленда, но стоит прослушать.', warn=True)
        elif corr_window < LOW_CORR:
            say(f'A и B слабо коррелированы ({corr_window:+.2f}) — это разные '
                f'по характеру IR; глубоких провалов при этом нет')
        print('Выравнивание и фаза')
        for line in lines:
            print(f'  {line}')
        print()
        blend = part_a + part_b

    # --- результат: обрезка, fade, нормализация — как в iroptimizator
    lines, say = block('результат')
    y, stats = process(blend, target_sr, target_sr, length, 0.0, 0.0,
                       norm=norm, loudness_target=loudness_target,
                       fix_polarity=a.fix_polarity)
    if stats is None:
        sys.exit('Ошибка: результат получился нулевым — A и B полностью погасили друг друга.')
    if stats['polarity'] == 'fixed':
        say('главный пик был отрицательный — полярность исправлена')
    elif stats['polarity'] == 'inverted':
        say('главный пик отрицательный — полярность инвертирована. '
            'Исправить: --fix-polarity.')
    say(f'в {length} сэмпл. уложилось {stats["kept"]:.2f} % энергии '
        f'{"свёртки" if serial else "смеси"} '
        f'(fade-out снял ещё {stats["lost_to_fade"]:.2f} %)')
    say(f'громкость {stats["loudness"]:.1f} dB (K-взвеш. энергия), '
        f'пик {stats["peak_dbfs"]:.1f} dBFS')
    if 'short_db' in stats:
        say(f'до целевой громкости не хватает {stats["short_db"]:.1f} dB: пик упёрся '
            f'в {PEAK_DBFS:g} dBFS.', warn=True)

    stem = blend_name(*(os.path.splitext(os.path.basename(f))[0] for f in a.files),
                      joiner='_x_' if serial else '+')
    ratio = '' if serial or mix == 50 else f'_{mix:g}-{100 - mix:g}'
    name = (f'{stem}{ratio}_{target_sr // 1000}k{bits}b_{length}'
            + ('_loud' if norm == 'loudness' else '') + '.wav')
    os.makedirs(a.outdir, exist_ok=True)
    out = os.path.join(a.outdir, name)
    if bits == 16:
        sf.write(out, quantize_pcm16(y, dither), target_sr, subtype='PCM_16')
    else:
        sf.write(out, y, target_sr, subtype=SUBTYPES[bits])
    say(f'→ {out}: {target_sr} Гц, {len(y)} сэмпл., {bits} бит'
        + (', TPDF-дизеринг' if dither else ''))
    print('Результат')
    for line in lines:
        print(f'  {line}')
    if warned:
        print('\nЕсть предупреждения — см. строки «ВНИМАНИЕ» выше.')


if __name__ == '__main__':
    main()
