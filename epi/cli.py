"""Командная строка пакета.

    python -m epi sync <папка>          # обойти папки и обновить каталог
    python -m epi stats                 # выжимка статистики
    python -m epi patient <ключ>        # всё об одном пациенте
    python -m epi index                 # разметить окна
    python -m epi index --gap-before 60 --gap-after 300   # с зазором у приступа
    python -m epi split                 # раздать пациентов по фолдам
    python -m epi export --fold train   # сохранить выборку в .npz
    python -m epi plot --type Seizure   # нарисовать маркеры в файлы .png

Те же действия доступны как обычный API: см. документацию пакета epi.
"""

import argparse
import json
import sys

from .catalog import DB_NAME, Catalog
from .stats import render, summary
from .windows import WindowSet


def _window_set(catalog, args):
    return WindowSet(catalog, length_sec=args.length, stride_sec=args.stride,
                     min_overlap_sec=args.min_overlap,
                     gap_before_sec=args.gap_before, gap_after_sec=args.gap_after,
                     preictal_sec=args.preictal, merge_gap_sec=args.merge_gap,
                     positives_per_event=args.positives_per_event, rate=args.rate)


def build_parser():
    parser = argparse.ArgumentParser(
        prog="python -m epi", description="Каталог и обучающая выборка ЭЭГ .e")
    parser.add_argument("--db", default=DB_NAME, help="файл каталога SQLite")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("sync", help="обойти папку и обновить каталог")
    p.add_argument("root", nargs="?", default=".")

    p = sub.add_parser("stats", help="выжимка статистики")
    p.add_argument("--json", dest="json_path", help="дополнительно сохранить в JSON")

    p = sub.add_parser("patient", help="всё об одном пациенте")
    p.add_argument("key", help="ключ пациента или его начало")

    for name, help_text in (("index", "разметить окна"),
                            ("split", "раздать пациентов по фолдам"),
                            ("export", "сохранить выборку в .npz")):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--length", type=float, default=10.0)
        p.add_argument("--stride", type=float, default=5.0)
        p.add_argument("--min-overlap", type=float, default=5.0)
        p.add_argument("--gap-before", type=float, default=0.0,
                       help="зазор перед приступом, с: окна ближе не берутся")
        p.add_argument("--gap-after", type=float, default=0.0,
                       help="зазор после приступа, с (постиктальный хвост)")
        p.add_argument("--preictal", type=float, default=None,
                       help="длина окна ожидания приступа, с: включает "
                            "разметку под задачу прогноза")
        p.add_argument("--merge-gap", type=float, default=0.0,
                       help="приступы ближе стольких секунд считать одним событием")
        p.add_argument("--positives-per-event", type=int, default=None,
                       help="сколько положительных окон брать с одного события")
        p.add_argument("--rate", type=float, default=256.0)
        if name == "split":
            p.add_argument("--seed", type=int, default=0)
            p.add_argument("--holdout", type=float, default=0.15)
        if name == "export":
            p.add_argument("--fold", default="train")
            p.add_argument("--n", type=int, default=1000)
            p.add_argument("--positive-ratio", type=float, default=0.5)
            p.add_argument("--hard-ratio", type=float, default=0.0,
                           help="доля фоновых окон, взятых вплотную к приступу")
            p.add_argument("--hard-span", type=float, default=600.0,
                           help="ширина полосы трудных окон, с (от края зазора)")
            p.add_argument("--max-flat", type=float, default=0.01)
            p.add_argument("--seed", type=int, default=0)
            p.add_argument("--out", default="training")

    p = sub.add_parser("plot", help="нарисовать маркеры в файлы .png")
    p.add_argument("--type", dest="kind")
    p.add_argument("--guid")
    p.add_argument("--n", type=int, default=2)
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--pad", type=float, default=5.0)
    p.add_argument("--window", type=float, default=25.0)
    p.add_argument("--out", default="plots")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    catalog = Catalog(args.db)

    if args.command == "sync":
        catalog.sync(args.root)

    elif args.command == "stats":
        report = summary(catalog)
        if args.json_path:
            with open(args.json_path, "w", encoding="utf-8") as handle:
                json.dump(report, handle, ensure_ascii=False, indent=2)
            print("записано в %s" % args.json_path)
        render(report)

    elif args.command == "patient":
        info = catalog.patient(args.key)
        print("пациент %s" % info["patient_key"])
        print("  alt_id %s | дата рождения %s | заметки: %s"
              % (info["alt_id"], info["dob"], (info["notes"] or "").strip()))
        for rec in info["recordings"]:
            print("\n  %s  %g Гц, %d каналов, %d сегментов, %.1f мин"
                  % (rec["folder"], rec["sampling_rate"], rec["n_channels"],
                     rec["n_segments"], rec["duration_sec"] / 60))
        for ev in info["seizures"]:
            print("      приступ  %s  сегмент %-3d на %8.1f с, длится %.1f с"
                  % (ev["folder"], ev["segment"], ev["segment_time_sec"],
                     ev["duration_sec"] or 0))

    elif args.command == "index":
        _window_set(catalog, args).build()

    elif args.command == "split":
        _window_set(catalog, args).split(seed=args.seed, holdout=args.holdout)

    elif args.command == "export":
        _window_set(catalog, args).export(
            fold=args.fold, out_dir=args.out, n=args.n,
            positive_ratio=args.positive_ratio, hard_ratio=args.hard_ratio,
            hard_span_sec=args.hard_span, max_flat=args.max_flat, seed=args.seed)

    elif args.command == "plot":
        if not args.kind and not args.guid:
            raise SystemExit("укажите --type или --guid")
        import matplotlib
        matplotlib.use("Agg")
        from pathlib import Path

        from .plotting import plot_marker

        events = catalog.events(type=args.kind, guid=args.guid,
                                limit=args.n, offset=args.offset)
        if not events:
            raise SystemExit("подходящих маркеров нет")
        out_dir = Path(args.out)
        out_dir.mkdir(exist_ok=True)
        for event in events:
            figure = plot_marker(catalog, event, pad_sec=args.pad,
                                 window_sec=args.window)
            name = "%s_%s_%d.png" % (event["folder"][:18], event["type"][:12],
                                     event["id"])
            path = out_dir / name.replace(" ", "_")
            figure.savefig(path, dpi=110)
            print(path)


if __name__ == "__main__":
    main(sys.argv[1:])
