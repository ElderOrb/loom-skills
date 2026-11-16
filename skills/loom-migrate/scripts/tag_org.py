#!/usr/bin/env python3
"""Tag the head of every `*-loom` model repo of a Hugging Face org (before an SDK migration's staging cards are promoted).

usage: tag_org.py --org ORG --tag TAG [--message MSG] [--only REPO ...] [--dry-run]

A repo that already has TAG is skipped (never moved). Per repo it prints: created / exists / would-create, and whether the
tag's commit is the repo's head. Needs huggingface_hub and a write token (HF_TOKEN or ~/.cache/huggingface/token).
"""
import argparse
import sys


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--org", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--message", default=None)
    ap.add_argument("--only", nargs="*", default=None, help="repo names (without the org) instead of every *-loom repo")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    from huggingface_hub import HfApi
    api = HfApi()
    names = a.only if a.only is not None else sorted(m.id.split("/", 1)[1] for m in api.list_models(author=a.org) if m.id.endswith("-loom"))
    bad = 0
    for name in names:
        repo = f"{a.org}/{name}"
        head = api.model_info(repo).sha
        tags = {t.name: t.target_commit for t in api.list_repo_refs(repo).tags}
        if a.tag in tags:
            at = api.model_info(repo, revision=a.tag).sha
            print(f"{repo}: exists ({a.tag} -> {at[:10]}{'' if at == head else ', not the head ' + head[:10]})")
            continue
        if a.dry_run:
            print(f"{repo}: would create {a.tag} at {head[:10]}")
            continue
        try:
            api.create_tag(repo, tag=a.tag, revision=head, tag_message=a.message or f"Cards built for Loom SDK {a.tag}")
            print(f"{repo}: created {a.tag} at {head[:10]}")
        except Exception as e:  # report and go on: one repo's failure must not hide the others
            bad += 1
            print(f"{repo}: FAILED {e}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
