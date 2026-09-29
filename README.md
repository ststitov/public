# public

Открытые проекты и форки. Лицензия — своя у каждого проекта, см. таблицу.

| Проект | Назначение | Лицензия |
|---|---|---|
| [IR-optimizator/](IR-optimizator/) | Подготовка импульсных характеристик (IR) гитарных кабинетов для аппаратных загрузчиков с ограничением длины (Pangaea CP-16M и подобные): ресемплинг, обрезка, fade-out, нормализация по пику или громкости, анализ исходника и верности конвертации; сводка по IR в PNG; объединение двух IR — бленд кабинетов или свёртка «усилитель → кабинет» | [MIT](IR-optimizator/LICENSE) |
| [tproxy-server/](tproxy-server/) | Веб-прокси для Telegram (Go, proof-of-concept) — копия форка [ststitov/tproxy-server](https://github.com/ststitov/tproxy-server) проекта [telegramdesktop/tproxy-server](https://github.com/telegramdesktop/tproxy-server), автор John Preston; перенесён с полной историей | не указана — все права у автора, см. [NOTICE](tproxy-server/NOTICE.md) |

Каждый проект автономен. IR-optimizator — Python, собственный `requirements.txt`, работа в виртуальном окружении:

```bash
cd IR-optimizator
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

tproxy-server — Go: `cd tproxy-server && go test ./... && go build -trimpath -o tproxy-server ./cmd/tproxy-server`.

Подробности — в `README.md` каждого проекта.
