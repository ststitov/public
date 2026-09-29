#!/usr/bin/env python3
"""
Адаптация IR кабинетов под Pangaea CP-16M и подобные устройства.

Параметры (частота дискретизации, битность, длина импульса, HPF, LPF)
запрашиваются интерактивно; Enter принимает значение по умолчанию
(48 кГц / 24 бит / 984 сэмпла / HPF и LPF выключены / нормализация по
пику). Любой параметр
можно задать флагом CLI — тогда соответствующий вопрос не задаётся;
при неинтерактивном вводе берутся значения по умолчанию.

Использование:
    python3 iroptimizator.py file1.wav [file2.wav ...] [-o outdir]
            [--sr 44|48] [--bits 16|24] [--len 512|984|1024|2048]
            [--hpf 60] [--lpf 8000] [--hpf-after]
            [--norm peak|loudness] [--loudness-target 8]
            [--fix-polarity] [--trim] [--no-dither] [--minphase]

Зависимости: pip install numpy scipy soundfile
"""
import argparse
import os
import sys
from math import gcd

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt, resample_poly

try:
    import soxr
except ImportError:                             # запасной путь без soxr
    soxr = None

SR_CHOICES = (44100, 48000)
BITS_CHOICES = (16, 24)
LEN_CHOICES = (512, 984, 1024, 2048)

DEFAULT_SR = 48000
DEFAULT_BITS = 24
DEFAULT_LEN = 984
# Фильтры по умолчанию выключены: в 984 сэмплах HPF не даёт чистого среза —
# обрезка возвращает низы (см. README, «HPF и короткое окно»).
DEFAULT_HPF_HZ = 0.0      # 0 — фильтр выключен
DEFAULT_LPF_HZ = 0.0      # 0 — фильтр выключен
HPF_ORDER = 2             # 12 дБ/окт
LPF_ORDER = 4             # 24 дБ/окт

FADE_MS = 2.0             # длительность fade-out, половина косинусного окна
PEAK_DBFS = -1.0          # пик при нормализации по пику и предел при нормализации по громкости
NORM_CHOICES = ('peak', 'loudness')
DEFAULT_NORM = 'peak'
# Цель нормализации по громкости: K-взвешенная энергия IR, дБ. Чуть ниже, чем у
# кабинетов с близким микрофоном при пике −1 dBFS (8.3–9.1 дБ): они почти не
# меняются, а IR с размазанной во времени энергией убавляются до их уровня.
DEFAULT_LOUDNESS_TARGET = 8.0
PREDELAY_THR_DB = -60.0   # порог детекта предзадержки относительно пика
BAND_THR_DB = -40.0       # порог полезной полосы относительно пика спектра
NATIVE_GAP_DB = 30.0      # провал спектра, по которому опознаётся обрыв
EDGE_OF_BAND = 0.85       # доля Найквиста файла, выше которой обрыв — край записи
CUTOFF_MARGIN = 0.9       # запас на точность детекта при сравнении с целью
ANTIPHASE_CORR = -0.5     # корреляция каналов, ниже которой это противофаза
LOW_CORR = 0.5            # корреляция, ниже которой свод даёт гребёнку
ENERGY_WARN_PCT = 99.0    # доля энергии в окне, ниже которой стоит предупредить
HPF_AFTER_MIN_GAIN = 0.7  # повторный HPF полезен, если энергия ниже среза падает сильнее
SR_CANDIDATES = (22050, 32000, 44100, 48000, 88200, 96000)

CLIP_RUN = 3              # столько отсчётов подряд на пике — признак клиппинга
DC_WARN_DB = -60.0        # постоянное смещение в хвосте выше этого — брак
NOISE_TAIL_SHARE = 0.1    # доля конца файла для оценки шумового пола
NOISE_TAIL_MIN_MS = 10.0  # но не короче: на коротком куске затухание неотличимо от шума
NOISE_PLATEAU_DB = 3.0    # хвост считается шумом, если уровень в нём не падает
NOISE_MARGIN_DB = 6.0     # импульс «в шуме», когда огибающая не выше пола + это
DIGITAL_SILENCE_DB = -120.0  # ниже этого от пика — цифровая тишина (дополнение нулями)
FIDELITY_LO_HZ = 100.0    # рабочая полоса оценки верности конвертации
FIDELITY_HI_HZ = 5000.0
FIDELITY_WARN_DB = 3.0    # отклонение АЧХ, выше которого тембр заметно изменён
LOW_BAND_LO_HZ = 40.0     # низ полосы «низов» — ниже рабочей полосы, до разрешения окна
LOW_BAND_REC_DB = 2.0     # отклонение на низах, при котором рекомендуется длина больше
LOW_BAND_SHOW_DB = 0.5    # подъёмы и провалы меньше этого в отчёт не выводятся

SUBTYPES = {16: 'PCM_16', 24: 'PCM_24'}
DITHER_SEED = 0           # фиксированное зерно: повторный запуск даёт тот же файл
# Порог рекомендации --minphase: минимально-фазовая версия должна заметно
# выигрывать. У кабинетов с близким микрофоном выигрыш ~0.01 п.п. (они и так
# почти минимально-фазовые), у IR с далёким микрофоном — единицы п.п.
MINPHASE_GAIN_PP = 1.0    # рост энергии в окне, процентных пунктов
MINPHASE_GAIN_DB = 1.0    # уменьшение отклонения АЧХ, дБ
SOXR_QUALITY = 'VHQ'


# ---------------------------------------------------------------- интерактив

def _batch_default(what, default_label):
    """Неинтерактивный запуск: сообщаем, какое значение по умолчанию взято."""
    print(f'{what}: ввод не интерактивный, используется значение '
          f'по умолчанию — {default_label.rstrip(".")}.')


def ask_choice(prompt, choices, default, labels=None):
    """Меню с нумерованными вариантами. Enter — значение по умолчанию."""
    labels = labels or [str(c) for c in choices]
    default_idx = choices.index(default)
    if not sys.stdin.isatty():
        _batch_default(prompt, labels[default_idx])
        return default
    print(f'\n{prompt}:')
    for i, label in enumerate(labels, 1):
        mark = '  (по умолчанию)' if i - 1 == default_idx else ''
        print(f'  {i}) {label}{mark}')
    while True:
        raw = input(f'Выбор [1-{len(choices)}, Enter = {default_idx + 1}]: ').strip()
        if not raw:
            return default
        if raw.isdigit() and 1 <= int(raw) <= len(choices):
            return choices[int(raw) - 1]
        print('Некорректный ввод, повторите.')


def _cutoff_label(hz):
    return 'выкл.' if hz == 0 else f'{hz:g} Гц'


def ask_cutoff(prompt, default, nyquist, required=False):
    """Частота среза в Гц. Enter — значение по умолчанию, 0 — фильтр выключен.

    required — фильтр обязателен (HPF при --hpf-after): ни Enter, ни 0
    не принимаются, нужно ввести частоту. Неинтерактивный запуск с
    required сюда не доходит — main() завершается с ошибкой раньше.
    """
    if not sys.stdin.isatty():
        _batch_default(prompt, _cutoff_label(default))
        return default
    if required:
        hint = 'обязательно при --hpf-after, например 80'
    elif default == 0:
        hint = 'Enter = выкл.'
    else:
        hint = f'Enter = {default:g}, 0 = выключить'
    while True:
        raw = input(f'\n{prompt} [{hint}]: ').strip()
        if not raw:
            if not required:
                return default
            print('Частота HPF обязательна: --hpf-after без HPF не имеет смысла.')
            continue
        try:
            value = float(raw.replace(',', '.'))
        except ValueError:
            print('Некорректный ввод, повторите.')
            continue
        if 0 < value < nyquist or (value == 0 and not required):
            return value
        if required:
            print(f'Нужна частота среза 0 < f < {nyquist:g} Гц: --hpf-after '
                  f'без HPF не имеет смысла.')
        else:
            print(f'Частота среза должна быть в диапазоне 0 < f < {nyquist:g} Гц '
                  f'(частота Найквиста) либо 0 для отключения фильтра.')


# ------------------------------------------------------------------ обработка

def resample(x, sr, target_sr):
    """Ресемплинг: soxr, если он установлен, иначе полифазный scipy.

    soxr (SoX Resampler) даёт честный обрыв на частоте Найквиста — измерено
    подавление алиасов около −190 дБ против −8…−16 дБ у resample_poly, чей
    антиалиасный фильтр размазан на 20–24 кГц при 48 → 44.1 кГц.
    """
    if soxr is not None:
        return soxr.resample(x, sr, target_sr, quality=SOXR_QUALITY)
    g = gcd(sr, target_sr)
    return resample_poly(x, target_sr // g, sr // g)


def resampler_name():
    return f'soxr {SOXR_QUALITY}' if soxr is not None else 'scipy resample_poly'


def detect_channel_conflict(x):
    """(корреляция, падение уровня при своде в дБ) или None для моно.

    Итоговый файл всегда моно, поэтому каналы усредняются. Если они в
    противофазе, усреднение их взаимно уничтожает — на выходе оказалась бы
    тишина без единой жалобы; если просто слабо коррелированы (два разнесённых
    микрофона), свод даёт гребёнчатую фильтрацию. И то и другое стоит увидеть.
    """
    if x.ndim < 2 or x.shape[1] < 2:
        return None
    corr = 1.0
    for i in range(x.shape[1]):
        for j in range(i + 1, x.shape[1]):
            a, b = x[:, i], x[:, j]
            if a.std() == 0 or b.std() == 0:
                continue
            corr = min(corr, float(np.corrcoef(a, b)[0, 1]))
    rms_ch = float(np.sqrt(np.mean(x ** 2, axis=0)).mean())
    rms_mono = float(np.sqrt(np.mean(x.mean(axis=1) ** 2)))
    drop_db = 20 * np.log10(rms_mono / rms_ch) if rms_ch > 0 and rms_mono > 0 else -np.inf
    return corr, drop_db


def detect_predelay(x, thr_db=PREDELAY_THR_DB):
    """Число сэмплов тишины в начале IR — до первого превышения порога от пика.

    Обрезка до целевой длины идёт от начала файла, поэтому предзадержка
    съедает полезный хвост импульса. Сама обработка её не трогает —
    детект только предупреждает.
    """
    a = np.abs(x)
    peak = a.max()
    if peak <= 0:
        return 0
    above = np.flatnonzero(a > peak * 10 ** (thr_db / 20))
    return int(above[0]) if above.size else 0


def _spectrum(x, sr, window):
    """Сглаженный спектр мощности в дБ относительно своего максимума.

    Два режима, потому что одно окно не годится для обеих задач:

    window=False — только короткий спад на хвосте. Атака импульса остаётся
    нетронутой, поэтому оценка полезной полосы честная (окно Ханна занижает
    её на пару килогерц, подавляя фронт).

    window=True — сигнал выравнивается по началу импульса и умножается на
    окно Ханна. Полоса при этом занижена, зато динамический диапазон
    достаточен, чтобы увидеть провал от антиалиасного фильтра. Выравнивание
    делает результат независимым от предзадержки.
    """
    x = np.asarray(x, dtype=np.float64)
    if window:
        a = np.abs(x)
        peak = a.max()
        if peak > 0:
            above = np.flatnonzero(a > peak * 10 ** (PREDELAY_THR_DB / 20))
            if above.size:
                x = x[int(above[0]):]
        x = x * np.hanning(len(x))
    else:
        tail = max(16, len(x) // 20)
        if len(x) > tail:
            x = x.copy()
            x[-tail:] *= 0.5 * (1 + np.cos(np.linspace(0, np.pi, tail)))
    n = 1 << max(10, (len(x) - 1).bit_length())
    power = np.abs(np.fft.rfft(x, n)) ** 2
    win = max(1, n // 512)                      # скользящее среднее по мощности
    power = np.convolve(power, np.ones(win) / win, mode='same')
    peak = power.max()
    if peak <= 0:                               # файл из одной тишины
        return np.fft.rfftfreq(n, 1 / sr), np.full_like(power, -np.inf)
    return np.fft.rfftfreq(n, 1 / sr), 10 * np.log10(power / peak + 1e-30)


def _band_mean_db(freqs, db, lo, hi):
    m = (freqs >= lo) & (freqs < hi)
    return float(db[m].mean()) if m.any() else None


def detect_content_cutoff(x, sr):
    """Частота, на которой содержание обрывается, или None — обрыва нет.

    Ресемплинг вверх оставляет над исходной частотой Найквиста провал от
    антиалиасного фильтра. Две оговорки, без которых детект врёт:

    Обрыв у самого края полосы файла (выше EDGE_OF_BAND от его Найквиста) не
    считается: там стоит антиалиасный фильтр АЦП любой честной записи — файл,
    записанный на 96 кГц, обрывается около 45 кГц просто потому, что так
    устроен АЦП.

    Полоса сравнения сверху ограничена 1.5·Найквиста кандидата, а не краем
    файла: иначе у широкополосного материала побеждает слишком низкий
    кандидат — пустота у верхнего края занижает среднее.

    Отсутствие обрыва не доказывает, что файл нативный: у гитарного IR
    содержание кончается на 10-19 кГц, и полосы по обе стороны от 22.05 кГц
    одинаково пусты — 44.1 и 48 кГц для такого материала неразличимы.
    """
    freqs, db = _spectrum(x, sr, window=True)
    nyq = sr / 2
    for cand in SR_CANDIDATES:
        c_nyq = cand / 2
        if c_nyq >= nyq * EDGE_OF_BAND:
            break
        below = _band_mean_db(freqs, db, c_nyq * 0.8, c_nyq * 0.98)
        above = _band_mean_db(freqs, db, c_nyq * 1.02, min(c_nyq * 1.5, nyq))
        if below is not None and above is not None and below - above >= NATIVE_GAP_DB:
            return c_nyq
    return None


def detect_band(freqs, db, thr_db=BAND_THR_DB):
    """Верхняя граница полезной полосы — последняя частота выше порога."""
    idx = np.flatnonzero(db > thr_db)
    return float(freqs[idx[-1]]) if idx.size else 0.0


def energy_below(x, sr, fc):
    """Доля энергии сигнала ниже частоты fc, в процентах."""
    n = 1 << max(17, (len(x) - 1).bit_length())
    power = np.abs(np.fft.rfft(x, n)) ** 2
    total = power.sum()
    if total <= 0:
        return 0.0
    return float(power[np.fft.rfftfreq(n, 1 / sr) < fc].sum() / total * 100)


def _apply_fade(x, target_sr):
    """Плавное затухание (half-cosine) на последних FADE_MS миллисекундах."""
    fade = min(int(round(FADE_MS / 1000 * target_sr)), len(x))
    if fade > 1:
        x[-fade:] *= 0.5 * (1 + np.cos(np.linspace(0, np.pi, fade)))
    return x


def hpf_after_corner(hpf_hz):
    """Частота −3 дБ после двух проходов HPF Баттерворта порядка HPF_ORDER.

    Один проход даёт −3 дБ на hpf_hz, два — уже −6 дБ, и точка −3 дБ
    уезжает вверх: |H|² = v/(1+v), v = (f/fc)^(2n); два прохода дают −3 дБ
    при |H|² = 1/√2, откуда v = (1/√2)/(1 − 1/√2) ≈ 2.414 и
    f = fc · v^(1/2n) — для 2-го порядка это 1.2465·fc (80 Гц → ~100 Гц).
    """
    v = (1 / np.sqrt(2)) / (1 - 1 / np.sqrt(2))
    return hpf_hz * v ** (1 / (2 * HPF_ORDER))


def octave_smooth(freqs, power, out_freqs, fraction):
    """Усреднение мощности в полосе 1/fraction октавы вокруг каждой частоты."""
    if fraction <= 0:
        return np.interp(out_freqs, freqs, power)
    half = 2 ** (1 / (2 * fraction))
    cum = np.concatenate([[0.0], np.cumsum(power)])
    lo = np.searchsorted(freqs, out_freqs / half, side='left')
    hi = np.searchsorted(freqs, out_freqs * half, side='right')
    hi = np.maximum(hi, lo + 1)
    return (cum[hi] - cum[lo]) / (hi - lo)


def detect_clipping(x):
    """Число серий из CLIP_RUN и более отсчётов подряд на уровне пика.

    Ищется плоская вершина на пике канала, а не превышение 0 dBFS: файл
    могли обрезать при записи, а потом нормализовать ниже нуля, и тогда
    срезанные вершины стоят на любом уровне.
    """
    channels = x.T if x.ndim > 1 else [x]
    runs = 0
    for ch in channels:
        a = np.abs(ch)
        peak = a.max()
        if peak <= 0:
            continue
        flags = np.concatenate([[0], (a >= peak * (1 - 1e-6)).astype(int), [0]])
        edges = np.diff(flags)
        lengths = np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)
        runs += int(np.sum(lengths >= CLIP_RUN))
    return runs


def noise_floor(x, sr, cut_ms):
    """Шумовой пол исходника и положение точки обрезки относительно него.

    Возвращает None для тишины, иначе словарь:
      floor_db   — уровень хвоста от пика (если пол не достигнут — уровень
                   в конце импульса);
      reached    — достигнут ли пол: хвост считается шумом, если его уровень
                   перестал падать (последние NOISE_TAIL_SHARE импульса не
                   тише предыдущего такого же куска больше чем на
                   NOISE_PLATEAU_DB);
      noise_ms   — момент, после которого огибающая не выше пола +
                   NOISE_MARGIN_DB (только если пол достигнут);
      cut_db     — уровень огибающей в точке обрезки;
      end        — длина импульса в отсчётах без хвоста цифровой тишины;
      silence_ms — длительность этого хвоста.

    Хвост цифровой тишины (экспорт фиксированной длины, дополнение нулями)
    отрезается до оценки: иначе его нулевой RMS давал «пол −600 dB, не
    достигнут», хотя импульс давно закончился.
    """
    peak = np.max(np.abs(x))
    if peak <= 0:
        return None
    audible = np.flatnonzero(np.abs(x) > peak * 10 ** (DIGITAL_SILENCE_DB / 20))
    end = int(audible[-1]) + 1
    silence_ms = (len(x) - end) / sr * 1000
    x = x[:end]
    # Куски сравнения не короче NOISE_TAIL_MIN_MS: на 2–3 мс затухающий хвост
    # падает меньше чем на NOISE_PLATEAU_DB и ошибочно принимается за шум.
    k = max(int(len(x) * NOISE_TAIL_SHARE), int(sr * NOISE_TAIL_MIN_MS / 1000))
    k = min(k, len(x))
    rms = lambda seg: np.sqrt(np.mean(seg ** 2))
    last = rms(x[-k:])
    floor_db = 20 * np.log10(last / peak + 1e-30)
    # Импульс короче трёх кусков — оценить пол нельзя, считаем его не достигнутым.
    reached = (len(x) >= 3 * k and last > 0
               and 20 * np.log10(rms(x[-2 * k:-k]) / last) <= NOISE_PLATEAU_DB)
    # Огибающая — скользящий RMS 5 мс в той же шкале, что и пол (от пика).
    win = max(1, min(int(sr * 0.005), len(x)))
    env = np.sqrt(np.convolve(x ** 2, np.ones(win) / win, mode='valid'))
    env_db = 20 * np.log10(env / peak + 1e-30)
    t_ms = (np.arange(len(env)) + win / 2) / sr * 1000
    result = {'floor_db': floor_db, 'reached': bool(reached), 'noise_ms': None,
              'cut_db': float(np.interp(cut_ms, t_ms, env_db)),
              'end': end, 'silence_ms': silence_ms}
    if reached:
        above = np.flatnonzero(env_db > floor_db + NOISE_MARGIN_DB)
        result['noise_ms'] = float(t_ms[above[-1]]) if above.size else 0.0
    return result


def _deviation_curve(result, reference, sr, f_lo, f_hi):
    """(сетка частот, отклонение АЧХ результата от эталона в дБ) без выравнивания."""
    n = 1 << max(17, (max(len(result), len(reference)) - 1).bit_length())
    freqs = np.fft.rfftfreq(n, 1 / sr)
    grid = np.logspace(np.log10(f_lo), np.log10(f_hi), 240)

    def smoothed_db(sig):
        power = octave_smooth(freqs, np.abs(np.fft.rfft(sig, n)) ** 2, grid, 3)
        return 10 * np.log10(power + 1e-30)

    return grid, smoothed_db(result) - smoothed_db(reference)


def conversion_fidelity(result, reference, sr, lo, hi):
    """Отклонение АЧХ результата от эталона в полосе [lo, hi].

    Эталон — исходник с теми же фильтрами на полной длине, поэтому в
    отклонение попадает только то, что внесли обрезка, fade и повторный
    HPF, а не намеренное действие фильтров. Общий уровень выравнивается
    по медиане: нормализация пика меняет громкость, а не тембр.
    Возвращает (макс. |отклонение| дБ, частота максимума, RMS отклонения).
    """
    grid, dev = _deviation_curve(result, reference, sr, lo, hi)
    dev -= np.median(dev)
    i = int(np.argmax(np.abs(dev)))
    return float(abs(dev[i])), float(grid[i]), float(np.sqrt(np.mean(dev ** 2)))


def low_band_deviation(result, reference, sr, low_lo, lo, hi):
    """Отклонение на низах [low_lo, lo) — ниже рабочей полосы верности.

    Ниже удвоенного разрешения окна форму АЧХ задаёт уже сама обрезка,
    поэтому основная оценка начинается выше. Но там же живут полка от
    незавершённого спада кабинета и резонанс корпуса (~50–60 Гц), которому
    не хватает времени прозвучать. Уровень выравнивается по рабочей полосе
    [lo, hi], а не по медиане самих низов: иначе равномерный провал всех
    низов «выровнялся» бы и исчез из оценки.
    Возвращает ((подъём дБ, частота), (провал дБ, частота)).
    """
    grid, dev = _deviation_curve(result, reference, sr, low_lo, hi)
    dev -= np.median(dev[grid >= lo])
    low = grid < lo
    g, d = grid[low], dev[low]
    up, down = int(np.argmax(d)), int(np.argmin(d))
    return (float(d[up]), float(g[up])), (float(d[down]), float(g[down]))


def quantize_pcm16(x, dither=True, seed=DITHER_SEED):
    """Квантование в 16 бит с округлением, результат — int16.

    dither — добавить TPDF-дизер: шум с треугольным распределением ±1 LSB
    (сумма двух равномерных) до округления. Без дизера ошибка квантования
    тихого сигнала коррелирована с ним и даёт дискретные искажения — на
    синусе в несколько LSB выбросы на 46–72 дБ над уровнем шума; с дизером
    ошибка становится ровным шумом около −96 dBFS.

    Квантование делается здесь, а не в libsndfile: она при записи float в
    int16 отбрасывает дробную часть, а не округляет (измерено: средняя
    ошибка −0.5 LSB, RMS 0.58 LSB вместо 0.29) — постоянное смещение на
    полразряда и шум на 6 дБ выше необходимого.
    """
    q = x * 32768
    if dither:
        rng = np.random.default_rng(seed)
        q = q + rng.uniform(-0.5, 0.5, len(x)) + rng.uniform(-0.5, 0.5, len(x))
    return np.clip(np.round(q), -32768, 32767).astype(np.int16)


def k_weighting_sos(fs):
    """K-фильтр ITU-R BS.1770 (основа LUFS) для произвольной частоты дискретизации.

    Два звена: полка +4 дБ выше ~1.7 кГц (влияние головы) и HPF ~38 Гц.
    Коэффициенты пересчитываются из аналоговых параметров, а не берутся
    табличными для 48 кГц — так фильтр верен и на 44.1 кГц.
    """
    gain_db, f0, q = 3.999843853973347, 1681.974450955533, 0.7071752369554196
    k = np.tan(np.pi * f0 / fs)
    vh = 10 ** (gain_db / 20)
    vb = vh ** 0.4996667741545416
    a0 = 1 + k / q + k * k
    shelf = [(vh + vb * k / q + k * k) / a0, 2 * (k * k - vh) / a0,
             (vh - vb * k / q + k * k) / a0, 1, 2 * (k * k - 1) / a0,
             (1 - k / q + k * k) / a0]
    f0, q = 38.13547087602444, 0.5003270373238773
    k = np.tan(np.pi * f0 / fs)
    a0 = 1 + k / q + k * k
    highpass = [1, -2, 1, 1, 2 * (k * k - 1) / a0, (1 - k / q + k * k) / a0]
    return np.array([shelf, highpass])


def ir_loudness(x, sr):
    """Громкость IR — K-взвешенная энергия в дБ (10·log10 Σ y², y — IR после K-фильтра).

    Для широкополосного сигнала на входе уровень на выходе свёртки
    пропорционален энергии IR, а K-фильтр учитывает чувствительность слуха.
    Нормализация по пику этого не видит: у IR с размазанной во времени
    энергией пик низкий, и пиковая нормализация делает его намного громче
    (на синтетическом «комнатном» импульсе — на 13.7 дБ).
    IR дополняется нулями на 200 мс, чтобы не потерять хвост самого фильтра.
    """
    padded = np.concatenate([x, np.zeros(int(sr * 0.2))])
    energy = np.sum(sosfilt(k_weighting_sos(sr), padded) ** 2)
    return float(10 * np.log10(energy)) if energy > 0 else -np.inf


def minimum_phase(x):
    """Минимально-фазовая версия импульса с той же АЧХ (гомоморфный метод).

    Из всех сигналов с данной АЧХ у минимально-фазового энергия максимально
    сдвинута к началу: для любой длины обрезки он сохраняет больше энергии.
    Строится через кепстр: log|X| → вещественный кепстр → свёртка причинной
    части → exp → обратное БПФ. Длина БПФ с 8-кратным запасом — против
    наложения во времени; пол −120 дБ — чтобы не брать log(0).
    Готовый scipy.signal.minimum_phase не подходит: он рассчитан на
    линейно-фазовые FIR и возвращает корень из их АЧХ.
    Знак результата от знака исходника не зависит (строится только по |X|).
    """
    n = 1 << max(10, (len(x) * 8 - 1).bit_length())
    mag = np.abs(np.fft.fft(x, n))
    peak = mag.max()
    if peak <= 0:
        return x.copy()
    cepstrum = np.fft.ifft(np.log(np.maximum(mag, peak * 1e-6))).real
    fold = np.zeros(n)
    fold[0] = 1
    fold[1:n // 2] = 2
    fold[n // 2] = 1
    return np.fft.ifft(np.exp(np.fft.fft(cepstrum * fold))).real[:len(x)]


def minphase_benefit(reference, sr, length, lo, hi):
    """Сравнение обрезки исходной и минимально-фазовой версий.

    Возвращает ((энергия в окне, %) исх., minphase, (отклонение АЧХ, дБ)
    исх., minphase); отклонения None, если полоса оценки пуста. Считается
    на отфильтрованном импульсе полной длины — том же эталоне, что и
    верность конвертации, — поэтому выигрыш прямой, а не косвенный.
    """
    mp = minimum_phase(reference)

    def kept(sig):
        return float(np.sum(sig[:length] ** 2) / np.sum(sig ** 2) * 100)

    def deviation(sig):
        if lo >= hi:
            return None
        cut = _apply_fade(np.pad(sig, (0, max(0, length - len(sig))))[:length].copy(), sr)
        return conversion_fidelity(cut, reference, sr, lo, hi)[0]

    return kept(reference), kept(mp), deviation(reference), deviation(mp)


def process(x, sr, target_sr, target_len, hpf_hz, lpf_hz, hpf_after=False,
            norm=DEFAULT_NORM, loudness_target=DEFAULT_LOUDNESS_TARGET,
            fix_polarity=False, trim=0, minphase=False):
    """Возвращает (обработанный сигнал, статистика) или (сигнал, None) для тишины.

    Статистика — словарь:
      kept          — % энергии отфильтрованного импульса, уложившейся в окно;
      lost_to_fade  — % энергии, снятый fade-out;
      below_hpf     — % энергии готового импульса ниже частоты HPF;
      below_hpf_pre — то же до повторного HPF (только при hpf_after);
      reference     — отфильтрованный импульс на полной длине до обрезки
                      (эталон для оценки верности конвертации);
      loudness      — громкость готового импульса (ir_loudness), дБ;
      peak_dbfs     — его пик;
      short_db      — на сколько не хватило до целевой громкости, если
                      пик упёрся в PEAK_DBFS (только norm='loudness').

    norm — 'peak': пик к PEAK_DBFS; 'loudness': громкость к loudness_target,
    но пик не выше PEAK_DBFS.

    fix_polarity — перевернуть импульс, если его главный пик отрицательный;
                   решение принимается по обработанному сигналу (после
                   фильтров и minphase), а не по исходнику: минимально-фазовая
                   версия строится только по АЧХ, и её знак от исходника не
                   зависит; stats['polarity'] — 'ok', 'inverted' или 'fixed';
    trim     — срезать столько отсчётов исходника в начале (предзадержка);
    minphase — заменить импульс минимально-фазовой версией перед обрезкой.

    Доли kept и lost_to_fade считаются от энергии отфильтрованного импульса
    на его полной длине: видно, сколько хвоста осталось за окном.

    hpf_after — повторить HPF после обрезки. Обрезка до короткого окна
    возвращает низы, которые HPF убрал на полной длине: это растекание
    полезного содержания из-за конечного окна. Повторный HPF их снова
    давит, но действует и на сам срез: фильтр становится вдвое круче,
    а точка −3 дБ смещается вверх (см. hpf_after_corner).
    """
    if x.ndim > 1:                       # стерео -> моно
        x = x.mean(axis=1)
    x = x.astype(np.float64, copy=False)
    if trim:                             # предзадержка — до ресемплинга, в отсчётах исходника
        x = x[trim:]
    if sr != target_sr:
        x = resample(x, sr, target_sr)
    # Фильтры применяются к ПОЛНОЙ длине IR (каузально, без пре-ринга),
    # чтобы их собственные хвосты не обрезались раньше времени
    if hpf_hz > 0:
        x = sosfilt(butter(HPF_ORDER, hpf_hz, 'highpass',
                           fs=target_sr, output='sos'), x)
    if lpf_hz > 0:
        x = sosfilt(butter(LPF_ORDER, lpf_hz, 'lowpass',
                           fs=target_sr, output='sos'), x)
    reference = x.copy()                 # эталон для оценок — до minphase
    if minphase:
        x = minimum_phase(x)
    polarity = 'ok'
    if x.size and x[int(np.argmax(np.abs(x)))] < 0:
        polarity = 'inverted'
        if fix_polarity:
            x = -x
            polarity = 'fixed'
    # Обрезка / дополнение до целевой длины
    energy_full = float(np.sum(x ** 2))
    x = np.pad(x, (0, max(0, target_len - len(x))))[:target_len]
    energy_window = float(np.sum(x ** 2))
    x = _apply_fade(x, target_sr)
    energy_faded = float(np.sum(x ** 2))

    stats = None
    if energy_full > 0:
        stats = {'kept': energy_window / energy_full * 100,
                 'lost_to_fade': (energy_window - energy_faded) / energy_full * 100,
                 'reference': reference, 'polarity': polarity}
        if hpf_hz > 0:
            if hpf_after:
                stats['below_hpf_pre'] = energy_below(x, target_sr, hpf_hz)
                # Хвост повторного фильтра за окном отрезается, поэтому fade
                # накладывается ещё раз — иначе на конце снова будет обрыв.
                x = _apply_fade(sosfilt(butter(HPF_ORDER, hpf_hz, 'highpass',
                                               fs=target_sr, output='sos'), x),
                                target_sr)
            stats['below_hpf'] = energy_below(x, target_sr, hpf_hz)

    # Нормализация — последней, после всех фильтров
    peak = np.max(np.abs(x))
    if peak > 0:
        peak_gain = 10 ** (PEAK_DBFS / 20) / peak
        if norm == 'loudness':
            gain = 10 ** ((loudness_target - ir_loudness(x, target_sr)) / 20)
            # Острый пик не должен уйти в клиппинг ради громкости: усиление
            # ограничивается пиком, недостача сообщается в отчёте.
            if gain > peak_gain:
                stats['short_db'] = 20 * np.log10(gain / peak_gain)
                gain = peak_gain
        else:
            gain = peak_gain
        x *= gain
        stats['loudness'] = ir_loudness(x, target_sr)
        stats['peak_dbfs'] = 20 * np.log10(np.max(np.abs(x)))
    return x, stats


# ----------------------------------------------------------------------- CLI

def parse_sr(value):
    """Принимает 44/48 (кГц) и 44100/48000 (Гц)."""
    table = {'44': 44100, '44.1': 44100, '48': 48000,
             '44100': 44100, '48000': 48000}
    if value not in table:
        raise argparse.ArgumentTypeError(
            'допустимые значения: 44, 48, 44100, 48000')
    return table[value]


def parse_bits(value):
    if value not in ('16', '24'):
        raise argparse.ArgumentTypeError('допустимые значения: 16, 24')
    return int(value)


def parse_len(value):
    try:
        n = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError('должно быть целым числом сэмплов')
    if n <= 0:
        raise argparse.ArgumentTypeError('должно быть больше нуля')
    return n


def parse_cutoff(value):
    try:
        f = float(value.replace(',', '.'))
    except ValueError:
        raise argparse.ArgumentTypeError('должно быть числом (Гц)')
    if f < 0:
        raise argparse.ArgumentTypeError('не может быть отрицательным')
    return f


def analyze_and_process(f, target_sr, length, hpf, lpf, hpf_after=False,
                        norm=DEFAULT_NORM, loudness_target=DEFAULT_LOUDNESS_TARGET,
                        fix_polarity=False, trim=False, minphase=False):
    """Анализ и обработка одного файла.

    Возвращает (сигнал, частота исходника, кадров на входе, строки отчёта,
    были ли предупреждения, были ли рекомендации). Отчёт собирается списком,
    а не печатается по ходу: при пакетной обработке он выводится одним
    блоком под именем файла.
    """
    lines = []
    warned = recommended = False

    def say(text, warn=False, recommend=False):
        nonlocal warned, recommended
        lines.append(f'ВНИМАНИЕ — {text}' if warn
                     else f'РЕКОМЕНДАЦИЯ — {text}' if recommend else text)
        warned = warned or warn
        recommended = recommended or recommend

    x, sr = sf.read(f)
    conflict = detect_channel_conflict(x)
    if conflict is not None:
        corr, drop_db = conflict
        drop_txt = 'полное' if drop_db == -np.inf else f'на {abs(drop_db):.1f} dB'
        if corr <= ANTIPHASE_CORR:
            say(f'каналы в противофазе (корреляция {corr:.2f}), при сведении '
                f'в моно они гасят друг друга: падение уровня {drop_txt}. '
                f'Проверьте полярность канала.', warn=True)
        elif corr < LOW_CORR:
            say(f'каналы слабо коррелированы (корреляция {corr:.2f}) — свод '
                f'в моно даёт гребёнчатую фильтрацию, уровень падает {drop_txt}')
        else:
            say(f'{x.shape[1]} канала сведены в моно (корреляция {corr:.2f})')

    mono = x.mean(axis=1) if x.ndim > 1 else x

    clips = detect_clipping(x)
    if clips:
        say(f'клиппинг в исходнике: срезанных вершин ({CLIP_RUN}+ отсчётов '
            f'подряд на пике) — {clips}. Вершины срезаны при записи или экспорте, '
            f'IR искажён, обработка это не исправит.', warn=True)
    predelay = detect_predelay(mono)
    trimmed = predelay if trim else 0
    # Точка обрезки на шкале исходника: при --trim окно начинается после
    # предзадержки, и сравнивать с хвостом нужно сдвинутую точку. Минимально-
    # фазовая версия тоже начинается сразу, без задержки, — сдвиг тот же.
    shift = predelay if (trim or minphase) else 0
    cut_ms = length / target_sr * 1000 + shift / sr * 1000
    nf = noise_floor(mono, sr, cut_ms)
    if nf is not None:
        end_ms = nf['end'] / sr * 1000
        # Обрезка за концом импульса: сравнивать её с хвостом нечего.
        past_end = cut_ms >= end_ms
        if nf['silence_ms'] >= 1.0:
            say(f'в конце файла {nf["silence_ms"]:.0f} мс цифровой тишины — '
                f'импульс заканчивается на {end_ms:.0f} мс')
        if not nf['reached']:
            say(f'шумовой пол не достигнут: импульс затухает до самого конца '
                f'(в конце {nf["floor_db"]:.0f} dB от пика)'
                + ('' if past_end else f'; в точке обрезки {cut_ms:.1f} мс '
                                       f'уровень {nf["cut_db"]:.0f} dB'))
        else:
            if past_end:
                where = 'окно длиннее импульса, за ним ничего нет'
            elif cut_ms < nf['noise_ms']:
                where = (f'обрезка на {cut_ms:.1f} мс (уровень {nf["cut_db"]:.0f} dB) '
                         f'раньше — хвост до ухода в шум теряется')
            else:
                where = f'обрезка на {cut_ms:.1f} мс позже — за окном только шум'
            say(f'шумовой пол {nf["floor_db"]:.0f} dB, импульс уходит в шум на '
                f'~{nf["noise_ms"]:.0f} мс; {where}')
            body = mono[:nf['end']]
            tail = body[-max(int(len(body) * NOISE_TAIL_SHARE),
                             int(sr * NOISE_TAIL_MIN_MS / 1000)):]
            dc_db = 20 * np.log10(abs(tail.mean()) / np.max(np.abs(mono)) + 1e-30)
            # Смещение значимо, если оно выше порога и заметно над случайным
            # средним шума (σ/√n), иначе это просто статистика шума.
            if dc_db > DC_WARN_DB and abs(tail.mean()) > 4 * tail.std() / np.sqrt(len(tail)):
                say(f'постоянное смещение в хвосте: {dc_db:.0f} dB от пика — '
                    f'похоже на смещение АЦП при записи. Уберите его HPF или '
                    f'в редакторе.', warn=True)

    freqs, db = _spectrum(mono, sr, window=False)
    band = detect_band(freqs, db)
    cutoff = detect_content_cutoff(mono, sr)
    verdict = ('обрывов спектра нет' if cutoff is None
               else f'содержание обрывается на {cutoff / 1000:.1f} кГц')
    say(f'полоса до {band / 1000:.1f} кГц '
        f'(порог {BAND_THR_DB:g} dB от пика спектра), {verdict}')
    if cutoff is not None and cutoff < target_sr / 2 * CUTOFF_MARGIN:
        say(f'содержание обрывается на {cutoff / 1000:.1f} кГц — ниже целевого '
            f'Найквиста {target_sr / 2 / 1000:.1f} кГц. Похоже, файл уже был '
            f'ресемплирован вверх; ресемплинг до {target_sr} Гц содержание '
            f'не вернёт.', warn=True)
    if lpf > 0 and band > 0 and lpf >= band:
        say(f'LPF {lpf:g} Гц выше полезной полосы ({band / 1000:.1f} кГц) — '
            f'фильтр практически ничего не срезает')

    if predelay and minphase and not trim:
        say(f'предзадержка {predelay} сэмпл. ({predelay / sr * 1000:.2f} мс) — при '
            f'--minphase уходит сама: минимально-фазовая версия начинается сразу')
    elif predelay and trim:
        say(f'предзадержка {predelay} сэмпл. ({predelay / sr * 1000:.2f} мс) удалена — '
            f'импульс начинается с первого отсчёта выше {PREDELAY_THR_DB:g} dB от пика')
    elif predelay:
        in_target = int(round(predelay / sr * target_sr))
        say(f'предзадержка {predelay} сэмпл. ({predelay / sr * 1000:.2f} мс, '
            f'порог {PREDELAY_THR_DB:g} dB от пика). Обрезка идёт от начала '
            f'файла: полезной части останется '
            f'{max(0, length - in_target)} из {length} сэмпл. Удалить её: --trim.',
            warn=True)

    if sr != target_sr:
        say(f'ресемплинг {sr} → {target_sr} Гц ({resampler_name()})')
        if soxr is None:
            say('soxr не установлен, используется resample_poly — его '
                'антиалиасный фильтр заметно положе. Поставьте soxr '
                '(pip install soxr) для лучшего подавления алиасов.', warn=True)
    y, stats = process(x, sr, target_sr, length, hpf, lpf, hpf_after,
                       norm, loudness_target, fix_polarity=fix_polarity,
                       trim=trimmed, minphase=minphase)
    if stats is None:
        say('сигнала в файле нет, на выходе тишина.', warn=True)
    else:
        if stats['polarity'] == 'fixed':
            say('главный пик был отрицательный — полярность исправлена, импульс перевёрнут')
        elif stats['polarity'] == 'inverted':
            say('главный пик отрицательный — полярность инвертирована. На звук '
                'одиночного IR не влияет, но при смешивании с другим IR или прямым '
                'сигналом даст гашение. Исправить: --fix-polarity.')
        kept = stats['kept']
        say(f'в {length} сэмпл. уложилось {kept:.2f} % энергии импульса '
            f'(fade-out снял ещё {stats["lost_to_fade"]:.2f} %)')
        if kept < ENERGY_WARN_PCT:
            say(f'за пределами окна осталось {100 - kept:.2f} % энергии, хвост '
                f'импульса обрезан заметно. Стоит взять длину больше '
                f'{length} сэмпл.', warn=True)
        # Рабочая полоса: снизу — не ниже удвоенного разрешения окна и
        # полутора частот HPF, сверху — ниже LPF и полезной полосы.
        lo = max(FIDELITY_LO_HZ, 2 * target_sr / length, 1.5 * hpf)
        hi = min(FIDELITY_HI_HZ, lpf / 1.5 if lpf > 0 else np.inf,
                 band if band > 0 else np.inf, target_sr / 2 * 0.9)
        # Минимально-фазовая версия: при --minphase — показать, что она дала;
        # без него — порекомендовать, если выигрыш заметный.
        k0, k1, d0, d1 = minphase_benefit(stats['reference'], target_sr, length, lo, hi)
        gain = (k1 - k0 >= MINPHASE_GAIN_PP
                or (d0 is not None and d1 is not None and d0 - d1 >= MINPHASE_GAIN_DB))
        effect = (f'энергия в окне {k0:.1f} → {k1:.1f} %'
                  + (f', отклонение АЧХ {d0:.1f} → {d1:.1f} dB' if d0 is not None else ''))
        if minphase:
            say(f'минимально-фазовое преобразование: {effect}'
                + ('' if gain else ' — на этом IR почти ничего не даёт, '
                                   'импульс и так близок к минимально-фазовому'))
        elif gain:
            say(f'включите --minphase: минимально-фазовая версия уложит в окно '
                f'больше импульса — {effect}. Фаза при этом меняется, временной '
                f'характер импульса может звучать иначе.', recommend=True)
        if lo < hi:
            worst, at, rms_dev = conversion_fidelity(y, stats['reference'],
                                                     target_sr, lo, hi)
            say(f'отклонение АЧХ от исходника в {lo:.0f} Гц–{hi / 1000:.1f} кГц: '
                f'макс. {worst:.1f} dB на {at:.0f} Гц, среднее {rms_dev:.1f} dB')
            if worst > FIDELITY_WARN_DB:
                say(f'обработка заметно меняет тембр: {worst:.1f} dB на {at:.0f} Гц. '
                    f'Обычно причина — слишком короткое окно для этого IR.',
                    warn=True)
        # Низы ниже рабочей полосы: полка от незавершённого спада кабинета и
        # недозвучавший резонанс корпуса. Нижняя граница — не ниже 1.5·HPF.
        low_lo = max(LOW_BAND_LO_HZ, 1.5 * hpf)
        if low_lo < lo < hi:
            (rise, f_rise), (drop, f_drop) = low_band_deviation(
                y, stats['reference'], target_sr, low_lo, lo, hi)
            parts = [f'подъём до {rise:+.1f} dB на {f_rise:.0f} Гц'
                     for _ in [0] if rise >= LOW_BAND_SHOW_DB]
            parts += [f'провал до {drop:+.1f} dB на {f_drop:.0f} Гц'
                      for _ in [0] if drop <= -LOW_BAND_SHOW_DB]
            say(f'низы {low_lo:.0f}–{lo:.0f} Гц (ниже рабочей полосы, ограничены '
                f'длиной окна): ' + (', '.join(parts) if parts
                                     else f'отклонение меньше {LOW_BAND_SHOW_DB:g} dB'))
            worst_low = max(rise, -drop)
            if worst_low > LOW_BAND_REC_DB:
                # Что даст следующая длина — на том же эталоне, без повторной
                # обработки: общий уровень всё равно выравнивается.
                longer = next((n for n in LEN_CHOICES if n > length), length * 2)
                cut = _apply_fade(np.pad(stats['reference'],
                                         (0, max(0, longer - len(stats['reference']))))[:longer].copy(),
                                  target_sr)
                (r2, _), (d2, _) = low_band_deviation(cut, stats['reference'], target_sr,
                                                      low_lo, lo, hi)
                say(f'на низах окно {length} сэмпл. заметно меняет тембр (до '
                    f'{worst_low:.1f} dB); при {longer} сэмпл. — до {max(r2, -d2):.1f} dB. '
                    f'Если загрузчик принимает более длинные IR — возьмите длину больше.',
                    recommend=True)
        if 'below_hpf_pre' in stats:
            pre, post = stats['below_hpf_pre'], stats['below_hpf']
            say(f'энергия ниже HPF {hpf:g} Гц: {pre:.2f} % → {post:.2f} % '
                f'(повторный HPF после обрезки; фактический срез −3 дБ '
                f'смещается с {hpf:g} до ~{hpf_after_corner(hpf):.0f} Гц)')
            # Критерий — заметный выигрыш, а не просто «не выросло»: у среза
            # рядом с пределом разрешения окна энергия чуть падает за счёт
            # полосы у самого среза, а самые низкие частоты становятся громче.
            if pre > 0 and post / pre > HPF_AFTER_MIN_GAIN:
                say(f'повторный HPF почти не помог: срез {hpf:g} Гц близок к '
                    f'пределу разрешения окна (~{target_sr / length:.0f} Гц), '
                    f'обрезка возвращает низы обратно, и самые низкие частоты '
                    f'могут стать даже громче. Флаг --hpf-after здесь лучше '
                    f'не использовать.', warn=True)
        elif 'below_hpf' in stats:
            say(f'энергия ниже HPF {hpf:g} Гц: {stats["below_hpf"]:.2f} %')
        if 'loudness' in stats:
            say(f'громкость {stats["loudness"]:.1f} dB (K-взвеш. энергия), '
                f'пик {stats["peak_dbfs"]:.1f} dBFS'
                + (f' — нормализовано по громкости, цель {loudness_target:g} dB'
                   if norm == 'loudness' else ''))
            if 'short_db' in stats:
                say(f'до целевой громкости {loudness_target:g} dB не хватает '
                    f'{stats["short_db"]:.1f} dB: пик упёрся в {PEAK_DBFS:g} dBFS. '
                    f'Файл тише остальных — у этого IR острый пик при малой '
                    f'энергии. Можно понизить --loudness-target.', warn=True)

    return y, sr, len(x), lines, warned, recommended


def main():
    ap = argparse.ArgumentParser(
        description='Адаптация IR кабинетов: ресемплинг, HPF/LPF, '
                    'обрезка до заданной длины, fade-out, нормализация.')
    ap.add_argument('files', nargs='+', help='исходные WAV-файлы')
    ap.add_argument('-o', '--outdir', default='.', help='каталог вывода')
    ap.add_argument('--sr', type=parse_sr,
                    help=f'частота дискретизации: 44 или 48 (кГц), '
                         f'по умолчанию {DEFAULT_SR // 1000}')
    ap.add_argument('--bits', type=parse_bits,
                    help=f'битность: 16 или 24, по умолчанию {DEFAULT_BITS}')
    ap.add_argument('--len', dest='length', type=parse_len,
                    help=f'длина импульса в сэмплах, по умолчанию {DEFAULT_LEN}')
    ap.add_argument('--hpf', type=parse_cutoff,
                    help=f'HPF, Гц (0 — выключить, по умолчанию {_cutoff_label(DEFAULT_HPF_HZ)})')
    ap.add_argument('--lpf', type=parse_cutoff,
                    help=f'LPF, Гц (0 — выключить, по умолчанию {_cutoff_label(DEFAULT_LPF_HZ)})')
    ap.add_argument('--hpf-after', action='store_true',
                    help='повторить HPF после обрезки: сильнее давит низы, '
                         'вернувшиеся из-за короткого окна, но срез становится '
                         'круче и точка −3 дБ смещается вверх примерно в 1.25 '
                         'раза (80 Гц → ~100 Гц)')
    ap.add_argument('--norm', choices=NORM_CHOICES,
                    help='нормализация: peak — пик к −1 dBFS (по умолчанию), '
                         'loudness — по громкости (K-взвешенная энергия IR), '
                         'чтобы IR разного характера звучали одинаково громко')
    ap.add_argument('--minphase', action='store_true',
                    help='заменить импульс минимально-фазовой версией с той же АЧХ: '
                         'в окно уложится больше энергии, но фаза меняется; '
                         'если это заметно поможет, скрипт сам порекомендует')
    ap.add_argument('--fix-polarity', action='store_true',
                    help='перевернуть импульс, если главный пик отрицательный')
    ap.add_argument('--trim', action='store_true',
                    help=f'удалить предзадержку — тишину до первого отсчёта выше '
                         f'{PREDELAY_THR_DB:g} dB от пика')
    ap.add_argument('--no-dither', action='store_true',
                    help='не применять TPDF-дизеринг при 16 битах (по умолчанию включён)')
    ap.add_argument('--loudness-target', type=float,
                    help=f'целевая громкость для --norm loudness, дБ '
                         f'(по умолчанию {DEFAULT_LOUDNESS_TARGET:g}); '
                         f'без --norm сама включает нормализацию по громкости')
    a = ap.parse_args()

    missing = [f for f in a.files if not os.path.isfile(f)]
    if missing:
        sys.exit('Файлы не найдены: ' + ', '.join(missing))

    # Имя результата строится из имени исходника без каталога, а суффикс у всех
    # файлов общий: a/cab.wav и b/cab.wav дали бы один выходной файл, и второй
    # молча затёр бы первый. Проверяется до вопросов и до обработки.
    stems = {}
    for f in a.files:
        stems.setdefault(os.path.splitext(os.path.basename(f))[0], []).append(f)
    clashes = {stem: fs for stem, fs in stems.items() if len(fs) > 1}
    if clashes:
        sys.exit('Ошибка: у разных исходников совпадёт имя результата, второй '
                 'затёр бы первый:\n' + '\n'.join(
                     f'  {stem}: ' + ', '.join(fs) for stem, fs in clashes.items())
                 + '\nПереименуйте файлы или обработайте их отдельными запусками '
                 'с разным -o.')

    # --hpf-after без HPF раньше молча игнорировался с кодом 0 — и скрипт,
    # ждущий файлы *_hpfpost.wav, получал обычные. Теперь это ошибка.
    if a.hpf_after and a.hpf == 0:
        sys.exit('Ошибка: --hpf-after требует включённого HPF, а задано --hpf 0.')
    if a.hpf_after and a.hpf is None and not sys.stdin.isatty():
        sys.exit('Ошибка: --hpf-after требует частоту HPF — укажите --hpf '
                 '(например, --hpf 80).')

    # Цель громкости без нормализации по громкости бессмысленна: с --norm peak
    # это ошибка, без --norm — выбор режима, и вопрос уже не нужен.
    if a.loudness_target is not None and a.norm == 'peak':
        sys.exit('Ошибка: --loudness-target задаёт цель нормализации по '
                 'громкости, а выбрано --norm peak.')
    if a.loudness_target is not None and a.norm is None:
        a.norm = 'loudness'
    loudness_target = (a.loudness_target if a.loudness_target is not None
                       else DEFAULT_LOUDNESS_TARGET)

    target_sr = a.sr or ask_choice(
        'Частота дискретизации', SR_CHOICES, DEFAULT_SR,
        [f'{sr // 1000}{"." + str(sr % 1000 // 100) if sr % 1000 else ""} кГц'
         for sr in SR_CHOICES])

    bits = a.bits or ask_choice('Битность', BITS_CHOICES, DEFAULT_BITS,
                                [f'{b} бит' for b in BITS_CHOICES])

    length = a.length or ask_choice(
        'Длина импульса', LEN_CHOICES, DEFAULT_LEN,
        [f'{n} сэмпл. ({n / target_sr * 1000:.1f} мс @ {target_sr} Гц)'
         for n in LEN_CHOICES])

    nyquist = target_sr / 2
    hpf = a.hpf if a.hpf is not None else ask_cutoff(
        'HPF, Гц', DEFAULT_HPF_HZ, nyquist, required=a.hpf_after)
    lpf = a.lpf if a.lpf is not None else ask_cutoff(
        'LPF, Гц', DEFAULT_LPF_HZ, nyquist)
    norm = a.norm or ask_choice(
        'Нормализация', NORM_CHOICES, DEFAULT_NORM,
        [f'по пику ({PEAK_DBFS:g} dBFS)',
         f'по громкости (цель {loudness_target:g} dB, пик не выше {PEAK_DBFS:g} dBFS)'])

    for name, value in (('HPF', hpf), ('LPF', lpf)):
        if value != 0 and not 0 < value < nyquist:
            sys.exit(f'Ошибка: {name} = {value:g} Гц вне диапазона '
                     f'0 < f < {nyquist:g} Гц (частота Найквиста для {target_sr} Гц).')

    hpf_after = a.hpf_after          # HPF при нём гарантированно > 0, см. выше
    dither = bits == 16 and not a.no_dither
    print(f'\nПараметры: {target_sr} Гц, {bits} бит, {length} сэмпл. '
          f'({length / target_sr * 1000:.1f} мс), '
          f'HPF {"выкл." if hpf == 0 else f"{hpf:g} Гц"}'
          f'{f" (+ повторно после обрезки, фактический срез ~{hpf_after_corner(hpf):.0f} Гц)" if hpf_after else ""}, '
          f'LPF {"выкл." if lpf == 0 else f"{lpf:g} Гц"}, '
          f'нормализация {f"по пику {PEAK_DBFS:g} dBFS" if norm == "peak" else f"по громкости {loudness_target:g} dB"}'
          + ''.join(f', {extra}' for extra, on in (
              ('минимальная фаза', a.minphase),
              ('исправление полярности', a.fix_polarity),
              ('удаление предзадержки', a.trim),
              ('TPDF-дизеринг', dither)) if on)
          + '\n')

    os.makedirs(a.outdir, exist_ok=True)
    print(f'Каталог вывода: {os.path.abspath(a.outdir)}\n')

    # Отдельные суффиксы, чтобы варианты с повторным HPF и нормализацией по
    # громкости не затирали обычный
    suffix = (f'_{target_sr // 1000}k{bits}b_{length}'
              + ('_hpfpost' if hpf_after else '')
              + ('_loud' if norm == 'loudness' else '')
              + ('_minph' if a.minphase else ''))
    total = len(a.files)
    warned_files, recommended_files = [], []
    for i, f in enumerate(a.files, 1):
        y, sr, frames, lines, warned, recommended = analyze_and_process(
            f, target_sr, length, hpf, lpf, hpf_after, norm, loudness_target,
            a.fix_polarity, a.trim, a.minphase)
        out = os.path.join(a.outdir,
                           os.path.splitext(os.path.basename(f))[0] + suffix + '.wav')
        if bits == 16:
            # 16 бит квантуются здесь (с округлением, при dither — с TPDF);
            # 24 бита — libsndfile: её отбрасывание дроби там даёт −144 dBFS.
            sf.write(out, quantize_pcm16(y, dither), target_sr, subtype='PCM_16')
        else:
            sf.write(out, y, target_sr, subtype=SUBTYPES[bits])
        lines.append(f'→ {os.path.basename(out)}: {target_sr} Гц, '
                     f'{len(y)} сэмпл., {bits} бит'
                     + (', TPDF-дизеринг' if dither else ''))

        header = os.path.basename(f) if total == 1 else f'[{i}/{total}] {os.path.basename(f)}'
        print(f'{header}  ({sr} Гц, {frames} сэмпл.)')
        for line in lines:
            print(f'  {line}')
        print()
        if warned:
            warned_files.append(os.path.basename(f))
        if recommended:
            recommended_files.append(os.path.basename(f))

    if total > 1:
        if warned_files:
            print(f'Обработано файлов: {total}, с предупреждениями '
                  f'{len(warned_files)}: ' + ', '.join(warned_files))
        else:
            print(f'Обработано файлов: {total}, предупреждений нет.')
        if recommended_files:
            print(f'С рекомендациями {len(recommended_files)}: '
                  + ', '.join(recommended_files))


if __name__ == '__main__':
    main()
