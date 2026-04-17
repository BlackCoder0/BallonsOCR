from __future__ import annotations

import traceback

from utils.asset_pack import ensure_pack_extracted, pack_path


def main() -> int:
    try:
        extracted = ensure_pack_extracted("core-models")
        if extracted:
            print(f"[bootstrap] extracted core asset pack: {pack_path('core-models')}")
        else:
            print("[bootstrap] core asset pack already ready.")
        return 0
    except Exception as exc:
        print(f"[bootstrap] failed to prepare core asset pack: {exc}")
        print(traceback.format_exc())
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
