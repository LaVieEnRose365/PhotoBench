import argparse
import json
import os
import time

from react_agent import ImageRetrievalAgent


def get_query_text(q, language):
    if language == "cn":
        return q.get("query_cn") or q.get("query") or q.get("query_en", "")
    if language == "en":
        return q.get("query_en") or q.get("query") or q.get("query_cn", "")
    return q.get("query_en") or q.get("query") or q.get("query_cn", "")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--face_info_path", required=True)
    parser.add_argument("--metadata_path", required=True)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--index_dir", required=True)
    parser.add_argument("--query_file", required=True)
    parser.add_argument("--output_folder", default=os.path.join(os.path.dirname(os.path.dirname(__file__)), "results"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--llm_model", default="gpt-4o-mini")
    parser.add_argument("--api_key", default=None)
    parser.add_argument("--base_url", default=None)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--concurrent", action="store_true")
    parser.add_argument("--max_workers", type=int, default=4)
    parser.add_argument("--language", choices=["cn", "en"], required=True)
    parser.add_argument("--tool_mode", default="ALL", choices=["V", "VM", "VF", "ALL"])
    args = parser.parse_args()

    out_dir = os.path.join(args.output_folder, args.language)
    os.makedirs(out_dir, exist_ok=True)
    output_file = os.path.join(out_dir, f"{args.model_name}_submission.json")

    agent = ImageRetrievalAgent(
        face_info_path=args.face_info_path,
        metadata_path=args.metadata_path,
        model_name=args.model_name,
        index_dir=args.index_dir,
        device=args.device,
        llm_model=args.llm_model,
        api_key=args.api_key,
        base_url=args.base_url,
        max_workers=args.max_workers,
        tool_mode=args.tool_mode,
    )

    with open(args.query_file, "r", encoding="utf-8") as f:
        queries = json.load(f)

    submissions = []
    start = time.time()

    if args.concurrent:
        query_texts = [get_query_text(q, args.language) for q in queries]
        all_preds, all_raw_counts = agent.query_batch(query_texts, max_workers=args.max_workers, return_raw_counts=True)
        for q, preds, raw_count in zip(queries, all_preds, all_raw_counts):
            original_preds = preds[:raw_count] if raw_count > 0 else []
            submissions.append({"query_id": q.get("Query_id") or q.get("query_id"), "predictions": original_preds[:100]})
    else:
        for q in queries:
            q_text = get_query_text(q, args.language)
            try:
                if args.verbose:
                    preds = agent.query(q_text, verbose=True)
                    raw_count = len(preds)
                else:
                    raw_count, preds = agent.query_with_raw_count(q_text)
            except Exception:
                preds = []
                raw_count = 0
            original_preds = preds[:raw_count] if raw_count > 0 else []
            submissions.append({"query_id": q.get("Query_id") or q.get("query_id"), "predictions": original_preds[:100]})

    elapsed = time.time() - start

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({"model_name": args.model_name, "language": args.language, "results": submissions}, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
