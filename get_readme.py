import json
from collections import defaultdict

# Define the categories and their corresponding tags in the desired output order.
ORDERED_CATEGORIES = [
    ("### Reinforced Video-LLMs", "RL"),
    ("### Video-LLM SFT for Reasoning", "SFT"),
    ("### Test-Time Scaling in Video Reasoning", "TTS"),
    ("### Benchmarks for Video Reasoning", "Benchmark"),
    ("### Related Surveys", "Survey")
]

# README header and introduction
README_HEADER = """# Awesome-Video-LLM-Post-Training [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

![image](./assets/timeline.png)

This Awesome list systematically curates and tracks the latest research in the post-training of Video-LLMs, with a special emphasis on works that enhance their reasoning capabilities. Following the taxonomy of the field, we focus on three key paradigms:


- 🧠 Reinforced Video-LLMs: Exploring how RL techniques are used to align Video-LLMs with human preferences or specific metrics. This includes methods like RLHF, DPO, GRPO and the design of effective reward models to enhance the logical consistency and factuality of model outputs.

- ⚙️ SFT for Reasoning: Collecting studies that leverage SFT on meticulously curated, reasoning-centric datasets. These works often incorporate CoT or other structured formats to directly teach models how to perform complex, multi-step reasoning.

- 🚀 Test-Time Scaling in Video Reasoning: Focusing on strategies that enhance reasoning capabilities at inference time without requiring further model training. This includes techniques like agentic frameworks, tool use, RAG, long CoT, and other methods that scale reasoning through computation.

- 📊 Benchmarks for Video Reasoning: Including the latest and most challenging benchmarks designed specifically to evaluate the complex reasoning abilities of Video-LLMs.

We hope this repository serves as a comprehensive and up-to-date resource hub for researchers and developers in this cutting-edge field. Contributions from the community are highly welcome via Pull Requests!

## Table of Contents

- [Awesome-Video-LLM-Post-Training ](#awesome-video-llm-post-training-)
    - [Table of Contents](#table-of-contents)
    - [Survey](#survey)
        - [Reinforced Video-LLMs](#reinforced-video-llms)
        - [Video-LLM SFT for Reasoning](#video-llm-sft-for-reasoning)
        - [Test-Time Scaling in Video Reasoning](#test-time-scaling-in-video-reasoning)
        - [Benchmarks for Video Reasoning](#benchmarks-for-video-reasoning)
        - [Related Surveys](#related-surveys)
    - [🌟 Star History](#-star-history)
    - [📝 Citation](#-citation)

    
![image](./assets/teaser.png)

## Survey
"""

# README footer
README_FOOTER = """


## 🌟 Star History

[![Star History Chart](https://api.star-history.com/svg?repos=yunlong10/Awesome-Video-LLM-Post-Training&type=Date)](https://star-history.com/#yunlong10/Awesome-Video-LLM-Post-Training&Date)


## 📝 Citation

```bibtex
@misc{tang2025videollmposttraining,
  title={Awesome Video-LLM Post-Training},
  year={2025},
  publisher={GitHub},
  howpublished={\\url{https://github.com/yunlong10/Awesome-Video-LLM-Post-Training}},
}
```
"""

def generate_markdown_table(papers):
    """Generates a Markdown table for a list of papers."""
    lines = []
    # Add table header
    lines.append("| Title | Paper | Code | Dataset | Venue |")
    lines.append("| :--- | :---: | :---: | :---: | :---: |")
    
    # Add a row for each paper
    for paper in papers:
        # Sanitize title to prevent breaking table format if it contains '|'
        title = paper.get('title', 'N/A').replace('|', '\\|')
        paper_link = paper.get('link', '#')
        
        # Get code link (if exists)
        code_link = paper.get('code', '')
        code_cell = f"[Link]({code_link})" if code_link else ""
        
        # Get dataset link (if exists)
        dataset_link = paper.get('dataset', '')
        dataset_cell = f"[Link]({dataset_link})" if dataset_link else ""
        
        # Get venue info
        venue = paper.get('venue', '')
        
        # Format the row
        row = f"| {title} | [Link]({paper_link}) | {code_cell} | {dataset_cell} | {venue} |"
        lines.append(row)
        
    return "\n".join(lines)

def main(json_file_path, output_md_path):
    """
    Reads paper data from a JSON file, categorizes and sorts the papers,
    and then generates a Markdown file with the results.
    """
    # 1. Read and parse the JSON file from the given path.
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            all_papers = json.load(f)
    except FileNotFoundError:
        print(f"Error: The file '{json_file_path}' was not found. Please make sure it exists.")
        return
    except json.JSONDecodeError:
        print(f"Error: Failed to parse JSON from the file '{json_file_path}'.")
        return

    # 2. Categorize papers based on their tags.
    categorized_papers = defaultdict(list)
    for paper in all_papers:
        for tag in paper.get('tags', []):
            categorized_papers[tag].append(paper)

    # 3. Sort papers within each category by release date (newest first).
    for tag in categorized_papers:
        # Use a default date string for papers that might be missing the 'released' key.
        categorized_papers[tag].sort(key=lambda p: p.get('released', '1970-01-01T00:00:00'), reverse=True)

    # 4. Generate the final Markdown content.
    markdown_output_parts = [README_HEADER]
    
    for header, tag_key in ORDERED_CATEGORIES:
        markdown_output_parts.append(header)
        markdown_output_parts.append("") # Add a blank line after the header
        
        papers_in_category = categorized_papers.get(tag_key, [])
        if papers_in_category:
            table_md = generate_markdown_table(papers_in_category)
            markdown_output_parts.append(table_md)
        else:
            # Add a placeholder if no papers are found for a category.
            markdown_output_parts.append("*No papers in this category yet.*")
        
        # Add extra spacing between sections for better readability.
        markdown_output_parts.append("")

    # Add the footer
    markdown_output_parts.append(README_FOOTER)

    # 5. Write the generated content to the output Markdown file.
    try:
        with open(output_md_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(markdown_output_parts))
        print(f"Successfully generated Markdown file at '{output_md_path}'")
    except IOError as e:
        print(f"Error writing to file '{output_md_path}': {e}")

if __name__ == '__main__':
    # This script depends on an existing 'papers.json' file in the same directory.
    input_json_filename = 'awesome_papers.json'
    output_md_filename = 'README.md'
    
    # Run the main function.
    main(input_json_filename, output_md_filename)