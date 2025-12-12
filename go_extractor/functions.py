"""
Convenience functions for GO extractor.
"""

import os
# Suppress tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from typing import Dict, Any, Optional
import pandas as pd
import numpy as np

def extract_go_embeddings(
    obo_path: str = "data/go-basic.obo",
    output_dir: str = "./output_simple",
    max_terms: Optional[int] = None,
    llm_model: Any = None,
    embedding_model_name: str = "nomic-embed-text",
    gemma_variant: str = "2b-it",
    gemma_weights_dir: Optional[str] = "/kaggle/input/gemma/transformers/1.1-2b-it/1/"
) -> pd.DataFrame:
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
        DataFrame with GO terms, embeddings, and metadata
    """
    print("🚀 Starting Simple GO Embeddings Extraction")
    print("=" * 50)
    print("✨ Features: No async complexity + Enhanced prompting + DataFrame output")

    extractor = SimpleGOExtractor(
        llm_model=llm_model,
        embedding_model_name=embedding_model_name,
        output_dir=output_dir,
        gemma_variant=gemma_variant,
        gemma_weights_dir=gemma_weights_dir
    )

    # Load GO data
    go_data = extractor.load_go_data(obo_path)
    if max_terms:
        go_data = go_data.head(max_terms)
        print(f"Limiting to {max_terms} terms for testing")

    # Generate enhanced descriptions and embeddings
    print("Generating enhanced descriptions for GO terms")
    descriptions = []
    for idx, row in go_data.iterrows():
        description = extractor.generate_description(row.to_dict())
        descriptions.append(description)
        if (idx + 1) % 1000 == 0:
            print(f"Generated descriptions for {idx + 1}/{len(go_data)} terms")

    # Generate embeddings for enhanced descriptions
    embeddings_enhanced = extractor.generate_embeddings(descriptions)

    # Generate embeddings for original OBO definitions
    print("Generating embeddings for original OBO definitions")
    original_definitions = go_data['definition'].tolist()
    embeddings_original = extractor.generate_embeddings(original_definitions)

    # Create comprehensive DataFrame
    df_results = pd.DataFrame({
        'go_id': go_data['go_id'].tolist(),
        'label': go_data['label'].tolist(),
        'namespace': go_data['namespace'].tolist(),
        'definition': go_data['definition'].tolist(),
        'synonyms': go_data['synonyms'].tolist(),
        'description_enhanced': descriptions,
        'embeddings_enhanced': list(embeddings_enhanced),
        'embeddings_original_definition': list(embeddings_original)
    })

    # Add metadata as attributes
    df_results.attrs['metadata'] = {
        'total_terms': len(go_data),
        'embedding_dim': embeddings_enhanced.shape[1],
        'embedding_model': "gemma" if hasattr(extractor, 'gemma_model') and extractor.gemma_model is not None else extractor.embedding_model_name,
        'llm_model': "gemma" if hasattr(extractor, 'gemma_model') and extractor.gemma_model is not None else (str(type(extractor.llm_model).__name__) if extractor.llm_model else "None"),
        'prompt_type': 'ml_optimized_structured',
        'namespace_distribution': go_data['namespace'].value_counts().to_dict()
    }

    # Save results
    extractor.save_results({
        'embeddings': embeddings_enhanced,
        'go_data': go_data,
        'go_ids': df_results['go_id'].tolist(),
        'labels': df_results['label'].tolist(),
        'descriptions': descriptions,
        'obo_definitions': original_definitions,
        'synonyms': df_results['synonyms'].tolist(),
        'namespaces': df_results['namespace'].tolist(),
        'metadata': df_results.attrs['metadata']
    })

    print(f"\n✅ Extraction complete!")
    print(f"📊 Processed {len(df_results)} GO terms")
    print(f"🔢 Embedding dimensions: {embeddings_enhanced.shape[1]}")

    namespace_dist = df_results.attrs['metadata']['namespace_distribution']
    print(f"📈 Namespace distribution:")
    for namespace, count in namespace_dist.items():
        print(f"   {namespace}: {count} terms")

    return df_results

def analyze_embeddings(df_results: pd.DataFrame):
    """Analyze embedding results from DataFrame."""
    print(f"\n📈 Embedding Analysis")
    print("=" * 30)

    metadata = df_results.attrs['metadata']
    embeddings_enhanced = np.array(df_results['embeddings_enhanced'].tolist())
    embeddings_original = np.array(df_results['embeddings_original_definition'].tolist())

    print(f"Total GO terms: {metadata['total_terms']}")
    print(f"Embedding dimensions: {metadata['embedding_dim']}")
    print(f"LLM used: {metadata['llm_model']}")

    print(f"\n📊 Enhanced Embedding Statistics:")
    print(f"  Shape: {embeddings_enhanced.shape}")
    print(f"  Mean: {embeddings_enhanced.mean():.6f}")
    print(f"  Std: {embeddings_enhanced.std():.6f}")
    print(f"  Min: {embeddings_enhanced.min():.6f}")
    print(f"  Max: {embeddings_enhanced.max():.6f}")

    print(f"\n📊 Original Definition Embedding Statistics:")
    print(f"  Shape: {embeddings_original.shape}")
    print(f"  Mean: {embeddings_original.mean():.6f}")
    print(f"  Std: {embeddings_original.std():.6f}")
    print(f"  Min: {embeddings_original.min():.6f}")
    print(f"  Max: {embeddings_original.max():.6f}")

    print(f"\n🔍 Sample GO Terms:")
    for i in range(min(5, len(df_results))):
        row = df_results.iloc[i]
        go_id = row['go_id']
        label = row['label']
        description = row['description_enhanced'][:200] + "..." if len(row['description_enhanced']) > 200 else row['description_enhanced']

        print(f"\n  {go_id}: {label}")
        print(f"    Enhanced Description: {description}")
        print(f"    Original Definition: {row['definition'][:200]}...")

def extract_simple_comparison(obo_path: str, max_terms: Optional[int] = None) -> pd.DataFrame:
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

    # Create DataFrame result
    df_results = pd.DataFrame({
        'go_id': go_data['go_id'].tolist(),
        'label': go_data['label'].tolist(),
        'namespace': go_data['namespace'].tolist(),
        'definition': go_data['definition'].tolist(),
        'synonyms': go_data['synonyms'].tolist(),
        'simple_descriptions': simple_descriptions,
        'embeddings_simple': list(embeddings)
    })

    # Add metadata as attributes
    df_results.attrs['metadata'] = {
        'total_terms': len(go_data),
        'embedding_dim': embeddings.shape[1],
        'prompt_type': 'simple_obo_definition',
        'namespace_distribution': go_data['namespace'].value_counts().to_dict()
    }

    extractor.save_results({
        'embeddings': embeddings,
        'go_data': go_data,
        'go_ids': df_results['go_id'].tolist(),
        'labels': df_results['label'].tolist(),
        'descriptions': simple_descriptions,
        'obo_definitions': df_results['definition'].tolist(),
        'synonyms': df_results['synonyms'].tolist(),
        'namespaces': df_results['namespace'].tolist(),
        'metadata': df_results.attrs['metadata']
    }, prefix="simple_go_embeddings")

    return df_results