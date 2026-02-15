import argparse
import json
import os

from image_search import ImageSearchEngine

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--index_dir", required=True)
    parser.add_argument("--query_file", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--instruction", default=None)
    parser.add_argument("--prompt_suffix", default="Represent the query for image retrieval.", type=str)
    parser.add_argument("--output_folder", default=os.path.join(os.path.dirname(os.path.dirname(__file__)), "results"))
    parser.add_argument("--language", required=True, choices=["cn", "en"])
    args = parser.parse_args()

    lang_output_folder = os.path.join(args.output_folder, args.language)
    os.makedirs(lang_output_folder, exist_ok=True)
    output_file = os.path.join(lang_output_folder, f"{args.model_name}_submission.json")

    engine = ImageSearchEngine.from_pretrained(model_name=args.model_name, index_dir=args.index_dir, device=args.device)
    engine.load()

    with open(args.query_file, "r", encoding="utf-8") as f:
        queries = json.load(f)

    submissions = []
    query_field = f"query_{args.language}"

    for q in queries:
        q_text = q[query_field]
        results = engine.retrieve(text=q_text, top_k=100, instruction=args.instruction, prompt_suffix=args.prompt_suffix)
        preds = [r["filename"] for r in results]
        submissions.append({"query_id": q.get("Query_id") or q.get("query_id"), "predictions": preds})

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({"model_name": args.model_name, "language": args.language, "results": submissions}, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
