"""
Convenience functions for GO extractor.
"""

import os
# Suppress tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from typing import Dict, Any, Optional

def extract_go_embeddings(
    obo_path: str = "data/go-basic.obo",
    output_dir: str = "./output_simple",
    max_terms: Optional[int] = None,
    llm_model: Any = None,
    embedding_model_name: str = "nomic-embed-text",
    gemma_variant: str = "2b-it",
    gemma_weights_dir: Optional[str] = "/kaggle/input/gemma/transformers/1.1-2b-it/1/"
) -> Dict[str, Any]:
    """
    Main function for GO embeddings extraction - simple synchronous version.
    Perfect for Jupyter notebooks and straightforward usage.

    Args:
        obo_path: Path to GO OBO file
        output_dir: Directory to save results
        max_terms: Maximum terms to process
        llm_model: LLM for description generation (optional)
        embedding_model_name: Name of embedding model
        gemma_variant: Gemma model variant (e.g., "2b-it", "7b-it")
        gemma_weights_dir: Path to Gemma model weights directory

    Returns:
        Dictionary with embeddings and metadata
    """
    print("🚀 Starting Simple GO Embeddings Extraction")
    print("=" * 50)
    print("✨ Features: No async complexity + Enhanced prompting")

    extractor = SimpleGOExtractor(
        llm_model=llm_model,
        embedding_model_name=embedding_model_name,
        output_dir=output_dir,
        gemma_variant=gemma_variant,
        gemma_weights_dir=gemma_weights_dir
    )

    results = extractor.extract_embeddings(obo_path, max_terms)
    extractor.save_results(results)

    print(f"\n✅ Extraction complete!")
    print(f"📊 Processed {results['metadata']['total_terms']} GO terms")
    print(f"🔢 Embedding dimensions: {results['metadata']['embedding_dim']}")

    namespace_dist = results['metadata']['namespace_distribution']
    print(f"📈 Namespace distribution:")
    for namespace, count in namespace_dist.items():
        print(f"   {namespace}: {count} terms")

    return results

def analyze_embeddings(results: Dict[str, Any]):
    """Analyze embedding results."""
    print(f"\n📈 Embedding Analysis")
    print("=" * 30)

    metadata = results['metadata']
    embeddings = results['embeddings']

    print(f"Total GO terms: {metadata['total_terms']}")
    print(f"Embedding dimensions: {metadata['embedding_dim']}")
    print(f"Prompt type: {metadata['prompt_type']}")
    print(f"LLM used: {metadata['llm_model']}")

    print(f"\n📊 Embedding Statistics:")
    print(f"  Shape: {embeddings.shape}")
    print(f"  Mean: {embeddings.mean():.6f}")
    print(f"  Std: {embeddings.std():.6f}")
    print(f"  Min: {embeddings.min():.6f}")
    print(f"  Max: {embeddings.max():.6f}")

    print(f"\n🔍 Sample Descriptions:")
    for i in range(min(5, len(results['go_ids']))):
        go_id = results['go_ids'][i]
        label = results['labels'][i]
        description = results['descriptions'][i][:200] + "..." if len(results['descriptions'][i]) > 200 else results['descriptions'][i]

        print(f"\n  {go_id}: {label}")
        print(f"    Description: {description}")

def extract_simple_comparison(obo_path: str, max_terms: Optional[int] = None) -> Dict[str, Any]:
    """Extract simple embeddings using just OBO definitions."""

    extractor = SimpleGOExtractor(llm_model=None)  # No LLM

    # Load GO data
    go_data = extractor.load_go_data(obo_path)
    if max_terms:
        go_data = go_data.head(max_terms)

    # Use simple descriptions (just OBO definitions)
    simple_descriptions = []
    for _, row in go_data.iterrows():
        simple_descriptions.append(f"GO term {row['go_id']}: {row['label']}. {row['definition']}")

    # Generate embeddings
    embeddings = extractor.generate_embeddings(simple_descriptions)

    results = {
        'embeddings': embeddings,
        'go_data': go_data,
        'go_ids': go_data['go_id'].tolist(),
        'labels': go_data['label'].tolist(),
        'simple_descriptions': simple_descriptions,
        'obo_definitions': go_data['definition'].tolist(),
        'metadata': {
            'total_terms': len(go_data),
            'embedding_dim': embeddings.shape[1],
            'prompt_type': 'simple_obo_definition',
            'namespace_distribution': go_data['namespace'].value_counts().to_dict()
        }
    }

    extractor.save_results(results, prefix="simple_go_embeddings")
    return results