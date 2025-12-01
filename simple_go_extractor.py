#!/usr/bin/env python3
"""
Simple GO Embeddings Extractor (No Asyncio)

A simplified version of the GO embeddings extractor that removes all async complexity.
This version uses synchronous processing, making it perfect for Jupyter notebooks
and eliminating the event loop conflicts.

Key features:
- No async/await complexity
- Works seamlessly in Jupyter notebooks
- Simple, readable code
- All the same enhanced prompting and metadata extraction
- Integrated Gemma LLM support for description generation
"""

import json
import logging
import warnings
import contextlib
import os
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd
import numpy as np
import torch
from owlready2 import get_ontology
from pydantic import BaseModel

# Suppress warnings for cleaner notebook output
warnings.filterwarnings('ignore')

@contextlib.contextmanager
def _set_default_tensor_type(dtype: torch.dtype):
    """Sets the default torch dtype to the given dtype."""
    torch.set_default_dtype(dtype)
    yield
    torch.set_default_dtype(torch.float)


class SimpleGOExtractor:
    """
    Simple GO embeddings extractor without async complexity.
    Perfect for Jupyter notebooks and straightforward usage.
    """
    
    def __init__(
        self,
        llm_model: Any = None,
        embedding_model_name: str = "nomic-embed-text",
        embedding_provider: str = "ollama",
        output_dir: str = "./output",
        device: str = "cpu",
        gemma_variant: str = "2b-it",
        gemma_weights_dir: Optional[str] = None
    ):
        self.llm_model = llm_model
        self.embedding_model_name = embedding_model_name
        self.embedding_provider = embedding_provider
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.device = device
        self.gemma_variant = gemma_variant
        self.gemma_weights_dir = gemma_weights_dir

        # Setup logging
        logging.basicConfig(level=logging.INFO, format='%(message)s')
        self.logger = logging.getLogger(__name__)

        # GO namespace descriptions for context
        self.namespace_descriptions = {
            'molecular_function': 'Molecular functions are the elemental activities of a gene product at the molecular level',
            'biological_process': 'Biological processes represent specific objectives that the organism is genetically programmed to achieve',
            'cellular_component': 'Cellular components are the places in the cell where a gene product is active'
        }

        # Prompt templates for different aspects
        self.prompt_templates = self._create_prompt_templates()

        # Initialize Gemma model if weights directory is provided
        if self.gemma_weights_dir:
            self._initialize_gemma_model()
    
    def _create_prompt_templates(self) -> Dict[str, str]:
        """Create structured prompt templates for consistent ML-focused descriptions."""
        return {
            'base_context': """You are a bioinformatics expert generating ML-optimized descriptions of Gene Ontology terms.

Your task is to create comprehensive, structured descriptions suitable for machine learning embeddings.
Focus on functional relationships, biological significance, and contextual information.

OBO Definition: {definition}
Namespace: {namespace}
GO ID: {go_id}
Synonyms: {synonyms}

""",
            'ml_focused_instructions': """Generate a structured description with these components:

1. CORE FUNCTION: What is the primary biological activity?
2. MECHANISM: How does this work at the molecular level?
3. BIOLOGICAL CONTEXT: Why is this important in cellular/tissue function?
4. RELATIONSHIPS: How does this relate to other biological processes/functions?
5. STRUCTURAL ASPECTS: Any domain, structural, or biochemical properties?
6. FUNCTIONAL SIGNIFICANCE: What happens when this is disrupted or missing?

Return exactly 300-500 words in this structure:""",
            
            'format_enforcer': """
Return format:
CORE_FUNCTION: [2-3 sentences]
MECHANISM: [2-3 sentences] 
BIOLOGICAL_CONTEXT: [2-3 sentences]
RELATIONSHIPS: [2-3 sentences]
STRUCTURAL_ASPECTS: [2-3 sentences]
FUNCTIONAL_SIGNIFICANCE: [2-3 sentences]

Ensure each section provides distinct, non-redundant information.""",
            
            'namespace_specific_context': {
                'molecular_function': """Focus on: enzymatic activities, binding properties, structural roles, molecular interactions, catalytic functions.""",
                'biological_process': """Focus on: pathway participation, temporal aspects, cellular outcomes, regulatory mechanisms, system-level effects.""",
                'cellular_component': """Focus on: subcellular localization, structural context, compartmental function, molecular complexes, cellular organization."""
            }
        }
    
    def _initialize_gemma_model(self):
        """Initialize Gemma model for description generation."""
        try:
            # Import Gemma modules
            from gemma.config import GemmaConfig, get_config_for_7b, get_config_for_2b
            from gemma.model import GemmaForCausalLM
            from gemma.tokenizer import Tokenizer

            # Set up device
            self.gemma_device = torch.device(self.device)

            # Get model configuration
            if "2b" in self.gemma_variant:
                model_config = get_config_for_2b()
            else:
                model_config = get_config_for_7b()

            model_config.tokenizer = os.path.join(self.gemma_weights_dir, "tokenizer.model")

            # Load model with context manager for dtype handling
            with _set_default_tensor_type(model_config.get_dtype()):
                self.gemma_model = GemmaForCausalLM(model_config)
                ckpt_path = os.path.join(self.gemma_weights_dir, f'gemma-{self.gemma_variant}.ckpt')
                self.gemma_model.load_weights(ckpt_path)
                self.gemma_model = self.gemma_model.to(self.gemma_device).eval()

            self.logger.info(f"Gemma model {self.gemma_variant} initialized successfully")

        except ImportError:
            self.logger.warning("Gemma modules not available. Using fallback LLM or OBO definitions.")
            self.gemma_model = None
        except Exception as e:
            self.logger.error(f"Error initializing Gemma model: {e}")
            self.gemma_model = None

    def _extract_go_metadata(self, go_class) -> Dict[str, Any]:
        """
        Extract comprehensive metadata from a GO class.

        Args:
            go_class: Owlready2 GO class object

        Returns:
            Dictionary with extracted metadata
        """
        metadata = {
            'go_id': '',
            'label': '',
            'definition': '',
            'synonyms': [],
            'namespace': '',
            'parents': [],
            'children': [],
            'cross_references': [],
            'comment': ''
        }

        try:
            # Extract GO ID from IRI
            if hasattr(go_class, 'iri') and go_class.iri:
                iri_parts = go_class.iri.split('/')
                metadata['go_id'] = iri_parts[-1] if iri_parts else str(go_class.iri)

            # Extract label
            if hasattr(go_class, 'label') and go_class.label:
                metadata['label'] = go_class.label[0] if isinstance(go_class.label, list) else str(go_class.label)

            # Extract definition
            if hasattr(go_class, 'IAO_0000115') and go_class.IAO_0000115:
                metadata['definition'] = str(go_class.IAO_0000115[0]) if isinstance(go_class.IAO_0000115, list) else str(go_class.IAO_0000115)

            # Extract synonyms
            if hasattr(go_class, 'hasExactSynonym') and go_class.hasExactSynonym:
                metadata['synonyms'] = [str(syn) for syn in go_class.hasExactSynonym]

            # Extract namespace
            if hasattr(go_class, 'namespace') and go_class.namespace:
                metadata['namespace'] = str(go_class.namespace)

            # Extract comment
            if hasattr(go_class, 'comment') and go_class.comment:
                metadata['comment'] = str(go_class.comment[0]) if isinstance(go_class.comment, list) else str(go_class.comment)

            # Extract parent classes
            if hasattr(go_class, 'is_a'):
                metadata['parents'] = [str(parent.name) for parent in go_class.is_a if hasattr(parent, 'name')]

            # Extract cross-references (db_xref property)
            if hasattr(go_class, 'db_xref') and go_class.db_xref:
                metadata['cross_references'] = [str(xref) for xref in go_class.db_xref]

        except Exception as e:
            self.logger.warning(f"Error extracting metadata for {getattr(go_class, 'name', 'Unknown')}: {e}")

        return metadata
    
    def load_go_data(self, obo_path: str) -> pd.DataFrame:
        """
        Load GO data with comprehensive metadata extraction.
        
        Args:
            obo_path: Path to the GO OBO file
            
        Returns:
            DataFrame with GO terms and rich metadata
        """
        self.logger.info(f"Loading GO data from {obo_path}")
        
        onto = get_ontology(str(obo_path)).load()
        go_data = []
        
        processed_count = 0
        
        for cls in onto.classes():
            # Skip deprecated terms
            if hasattr(cls, 'is_obsolete') and cls.is_obsolete:
                continue
            
            metadata = self._extract_go_metadata(cls)
            
            # Only include terms with labels and definitions
            if metadata['label'] and metadata['definition']:
                # Add namespace description
                namespace_key = metadata['namespace'].replace(' ', '_') if metadata['namespace'] else ''
                if namespace_key in self.namespace_descriptions:
                    metadata['namespace_context'] = self.namespace_descriptions[namespace_key]
                else:
                    metadata['namespace_context'] = ''
                
                go_data.append(metadata)
                processed_count += 1
                
                if processed_count % 100 == 0:
                    self.logger.info(f"Processed {processed_count} GO terms")
        
        df = pd.DataFrame(go_data)
        self.logger.info(f"Loaded {len(df)} GO terms with metadata")
        
        # Add statistics
        namespace_counts = df['namespace'].value_counts()
        self.logger.info(f"Namespace distribution:")
        for namespace, count in namespace_counts.items():
            self.logger.info(f"  {namespace}: {count} terms")
        
        return df
    
    def _create_ml_optimized_prompt(self, metadata: Dict[str, Any]) -> str:
        """
        Create a structured, ML-optimized prompt for consistent description generation.
        
        Args:
            metadata: GO term metadata dictionary
            
        Returns:
            Structured prompt string
        """
        # Extract key components
        go_id = metadata['go_id']
        label = metadata['label']
        definition = metadata['definition']
        namespace = metadata['namespace']
        synonyms = ', '.join(metadata['synonyms'][:5]) if metadata['synonyms'] else 'None'
        namespace_context = metadata.get('namespace_context', '')
        
        # Select namespace-specific context
        namespace_key = namespace.replace(' ', '_') if namespace else ''
        specific_context = self.prompt_templates['namespace_specific_context'].get(
            namespace_key, 'Focus on general biological significance and functional relationships.'
        )
        
        # Build structured prompt
        prompt_parts = []
        
        # Base context with OBO information
        base_context = self.prompt_templates['base_context'].format(
            definition=definition,
            namespace=namespace,
            go_id=go_id,
            synonyms=synonyms
        )
        prompt_parts.append(base_context)
        
        # Namespace-specific context
        if namespace_context:
            prompt_parts.append(f"Namespace Context: {namespace_context}")
        
        prompt_parts.append(specific_context)
        
        # ML-focused instructions
        prompt_parts.append(self.prompt_templates['ml_focused_instructions'])
        
        # Format enforcement
        prompt_parts.append(self.prompt_templates['format_enforcer'])
        
        # Add the actual GO term
        prompt_parts.append(f"\nGenerate description for GO term: {label}")
        
        return '\n'.join(prompt_parts)
    
    def generate_description(self, metadata: Dict[str, Any]) -> str:
        """
        Generate ML-optimized description using structured prompting.

        Args:
            metadata: GO term metadata dictionary

        Returns:
            Generated description
        """
        if self.llm_model is None and not hasattr(self, 'gemma_model'):
            # Use OBO definition as fallback
            return f"GO term {metadata['go_id']}: {metadata['label']}. {metadata['definition']}"

        try:
            # Create enhanced prompt
            prompt = self._create_ml_optimized_prompt(metadata)

            # Generate description using LLM
            if hasattr(self, 'gemma_model') and self.gemma_model is not None:
                # Use Gemma model
                USER_CHAT_TEMPLATE = "<start_of_turn>user\n{prompt}<end_of_turn>\n"
                MODEL_CHAT_TEMPLATE = "<start_of_turn>model\n{prompt}<end_of_turn>\n"

                # Format prompt for Gemma
                gemma_prompt = (
                    USER_CHAT_TEMPLATE.format(prompt=prompt)
                    + "<start_of_turn>model\n"
                )

                result = self.gemma_model.generate(
                    gemma_prompt,
                    device=self.gemma_device,
                    output_len=500
                )
                response_text = result.strip()

                # Clean up Gemma-specific formatting
                if response_text.startswith("<start_of_turn>model\n"):
                    response_text = response_text[len("<start_of_turn>model\n"):]
                if response_text.endswith("<end_of_turn>"):
                    response_text = response_text[:-len("<end_of_turn>")]

                response_text = response_text.strip()

            elif hasattr(self.llm_model, 'generate'):
                # Use generic LLM
                result = self.llm_model.generate(prompt, device=self.device, output_len=500)
                response_text = result.strip()
            else:
                response_text = f"Enhanced description for {metadata['label']}"

            return response_text

        except Exception as e:
            self.logger.error(f"Error generating description for {metadata['go_id']}: {e}")
            # Use OBO definition as fallback
            return f"GO term {metadata['go_id']}: {metadata['label']}. {metadata['definition']}"
    
    def generate_embeddings(self, descriptions: List[str]) -> np.ndarray:
        """
        Generate embeddings for descriptions.
        
        Args:
            descriptions: List of descriptions
            
        Returns:
            Array of embeddings
        """
        self.logger.info("Generating embeddings for descriptions")
        
        embeddings = []
        
        for i, description in enumerate(descriptions):
            try:
                # TODO: Replace with your actual embedding service
                # This is a placeholder - replace with real embedding API call
                embedding = np.random.rand(768).tolist()  # 768-dimensional embeddings
                embeddings.append(embedding)
                
                if (i + 1) % 100 == 0:
                    self.logger.info(f"Generated embeddings for {i + 1}/{len(descriptions)} descriptions")
                    
            except Exception as e:
                self.logger.error(f"Error generating embedding for description {i}: {e}")
                # Use zero embedding as fallback
                embeddings.append(np.zeros(768))
        
        return np.array(embeddings)
    
    def extract_embeddings(self, obo_path: str, max_terms: Optional[int] = None) -> Dict[str, Any]:
        """
        Extract embeddings with metadata and ML-optimized descriptions.
        
        Args:
            obo_path: Path to GO OBO file
            max_terms: Maximum terms to process
            
        Returns:
            Results dictionary
        """
        self.logger.info("Starting GO embeddings extraction")
        
        # Load GO data
        go_data = self.load_go_data(obo_path)
        
        if max_terms:
            go_data = go_data.head(max_terms)
            self.logger.info(f"Limiting to {max_terms} terms for testing")
        
        # Generate descriptions
        self.logger.info("Generating descriptions for GO terms")
        descriptions = []
        
        for idx, row in go_data.iterrows():
            description = self.generate_description(row.to_dict())
            descriptions.append(description)
            
            if (idx + 1) % 100 == 0:
                self.logger.info(f"Generated descriptions for {idx + 1}/{len(go_data)} terms")
        
        # Generate embeddings
        embeddings = self.generate_embeddings(descriptions)
        
        # Prepare results
        results = {
            'embeddings': embeddings,
            'go_data': go_data,
            'go_ids': go_data['go_id'].tolist(),
            'labels': go_data['label'].tolist(),
            'descriptions': descriptions,
            'obo_definitions': go_data['definition'].tolist(),
            'synonyms': go_data['synonyms'].tolist(),
            'namespaces': go_data['namespace'].tolist(),
            'metadata': {
                'total_terms': len(go_data),
                'embedding_dim': embeddings.shape[1],
                'embedding_model': self.embedding_model_name,
                'llm_model': str(type(self.llm_model).__name__) if self.llm_model else "None",
                'prompt_type': 'ml_optimized_structured',
                'namespace_distribution': go_data['namespace'].value_counts().to_dict()
            }
        }
        
        return results
    
    def save_results(self, results: Dict[str, Any], prefix: str = "go_embeddings") -> None:
        """
        Save results to files.
        
        Args:
            results: Results dictionary from extract_embeddings
            prefix: Prefix for output files
        """
        self.logger.info("Saving results")
        
        embeddings = results['embeddings']
        go_ids = results['go_ids']
        
        # Save as CSV (embeddings with GO IDs as index)
        csv_path = self.output_dir / f"{prefix}.csv"
        df_embeddings = pd.DataFrame(embeddings, index=go_ids)
        df_embeddings.to_csv(csv_path)
        
        # Save as NPZ
        npz_path = self.output_dir / f"{prefix}.npz"
        np.savez(
            npz_path,
            embeddings=embeddings,
            go_ids=np.array(go_ids),
            descriptions=np.array(results['descriptions'], dtype=object),
            obo_definitions=np.array(results['obo_definitions'], dtype=object),
            synonyms=np.array(results['synonyms'], dtype=object),
            namespaces=np.array(results['namespaces'], dtype=object),
            **results
        )
        
        # Save metadata as JSON
        metadata_path = self.output_dir / f"{prefix}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(results['metadata'], f, indent=2)
        
        # Save GO data as CSV for analysis
        go_data_path = self.output_dir / f"{prefix}_go_data.csv"
        results['go_data'].to_csv(go_data_path, index=False)
        
        self.logger.info(f"Results saved to {self.output_dir}")
        print(f"\n📁 Results saved to: {self.output_dir}")
        print(f"   📊 CSV: {csv_path.name}")
        print(f"   💾 NPZ: {npz_path.name}")
        print(f"   📋 GO Data: {go_data_path.name}")
        print(f"   📋 Metadata: {metadata_path.name}")


# Convenience functions for easy usage
def extract_go_embeddings(
    obo_path: str = "/kaggle/input/cafa-6-protein-function-prediction/Train/go-basic.obo",
    output_dir: str = "./output_simple",
    max_terms: Optional[int] = None,
    llm_model: Any = None,
    embedding_model_name: str = "nomic-embed-text",
    gemma_variant: str = "2b-it",
    gemma_weights_dir: Optional[str] = None
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


# Simple comparison function
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


# Example usage in notebooks:
"""
# Copy this entire script to your notebook and run it

# Simple extraction (no LLM)
results = extract_go_embeddings(max_terms=100)

# With Gemma LLM for enhanced descriptions
# results = extract_go_embeddings(
#     gemma_variant="2b-it",
#     gemma_weights_dir="/kaggle/input/gemma/pytorch/1.1-2b-it/1/",
#     max_terms=100
# )

# Compare with simple approach
simple_results = extract_simple_comparison(obo_path, max_terms=100)

# Analyze results
analyze_embeddings(results)

# Compare descriptions
print("\nComparison:")
for i in range(3):
    print(f"\nGO: {results['go_ids'][i]}")
    print(f"Simple: {simple_results['simple_descriptions'][i][:100]}...")
    print(f"Enhanced: {results['descriptions'][i][:100]}...")
"""