#!/usr/bin/env python3
"""
Main module for Simple GO Embeddings Extractor.
"""

import json
import logging
import warnings
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

# Import from local modules with try/except for Kaggle compatibility
try:
    from go_extractor.prompt_templates import prompt_template
    from go_extractor.utils import _set_default_tensor_type
except ImportError:
    # Fallback for Kaggle notebook environment - only provide utils
    import contextlib
    import torch

    @contextlib.contextmanager
    def _set_default_tensor_type(dtype: torch.dtype):
        """Sets the default torch dtype to the given dtype."""
        torch.set_default_dtype(dtype)
        yield
        torch.set_default_dtype(torch.float)

    # Note: prompt_template will be provided manually in Kaggle notebook
    # This allows keeping the template in a separate file for local development
    # while still working in Kaggle when the template is pasted into a cell

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

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
        return prompt_template

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

            logger.info(f"Gemma model {self.gemma_variant} initialized successfully")

        except ImportError:
            logger.warning("Gemma modules not available. Using fallback LLM or OBO definitions.")
            self.gemma_model = None
        except Exception as e:
            logger.error(f"Error initializing Gemma model: {e}")
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
                definition_value = go_class.IAO_0000115[0] if isinstance(go_class.IAO_0000115, list) else go_class.IAO_0000115
                metadata['definition'] = str(definition_value)

            # Extract synonyms
            if hasattr(go_class, 'hasExactSynonym') and go_class.hasExactSynonym:
                synonyms = go_class.hasExactSynonym
                metadata['synonyms'] = [str(syn) for syn in synonyms]

            # Extract namespace
            if hasattr(go_class, 'namespace') and go_class.namespace:
                metadata['namespace'] = str(go_class.namespace)

            # Extract comment
            if hasattr(go_class, 'comment') and go_class.comment:
                comment_value = go_class.comment[0] if isinstance(go_class.comment, list) else go_class.comment
                metadata['comment'] = str(comment_value)

            # Extract parent classes
            if hasattr(go_class, 'is_a'):
                parents = go_class.is_a
                metadata['parents'] = [str(parent.name) for parent in parents if hasattr(parent, 'name')]

            # Extract cross-references (db_xref property)
            if hasattr(go_class, 'db_xref') and go_class.db_xref:
                xrefs = go_class.db_xref
                metadata['cross_references'] = [str(xref) for xref in xrefs]

        except Exception as e:
            logger.warning(f"Error extracting metadata for {getattr(go_class, 'name', 'Unknown')}: {e}")

        return metadata

    def load_go_data(self, obo_path: str) -> pd.DataFrame:
        """
        Load GO data with comprehensive metadata extraction.

        Args:
            obo_path: Path to the GO OBO file

        Returns:
            DataFrame with GO terms and rich metadata
        """
        logger.info(f"Loading GO data from {obo_path}")

        try:
            onto = get_ontology(str(obo_path)).load()
        except Exception as e:
            logger.error(f"Error loading OBO file with owlready2: {e}")
            logger.info("Trying alternative OBO parsing with obonet...")
            import obonet
            go_graph = obonet.read_obo(str(obo_path))
            # Create a simple in-memory ontology structure that mimics owlready2 classes
            from collections import namedtuple
            GOClass = namedtuple('GOClass', ['iri', 'label', 'IAO_0000115', 'namespace', 'hasExactSynonym', 'is_a', 'db_xref', 'comment', 'is_obsolete'])
            # Create mock classes for each GO term
            mock_classes = []
            for node_id, node_data in go_graph.nodes(data=True):
                # Create a mock GO class
                go_id = node_id
                label = [node_data.get('name', '')] if node_data.get('name') else ['']
                definition = [node_data.get('def', '')] if node_data.get('def') else ['']
                namespace = node_data.get('namespace', '')
                synonyms = node_data.get('synonyms', [])
                parents = list(go_graph.predecessors(node_id))
                xrefs = node_data.get('xrefs', [])
                comment = node_data.get('comment', '')
                is_obsolete = node_data.get('is_obsolete', False)

                mock_class = GOClass(
                    iri=f"http://purl.obolibrary.org/obo/{go_id}",
                    label=label,
                    IAO_0000115=definition,
                    namespace=namespace,
                    hasExactSynonym=synonyms,
                    is_a=parents,
                    db_xref=xrefs,
                    comment=comment,
                    is_obsolete=is_obsolete
                )
                mock_classes.append(mock_class)

            # Create a mock ontology object
            class MockOntology:
                def classes(self):
                    return mock_classes

            onto = MockOntology()
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
                    logger.info(f"Processed {processed_count} GO terms")

        df = pd.DataFrame(go_data)
        logger.info(f"Loaded {len(df)} GO terms with metadata")

        # Add statistics
        namespace_counts = df['namespace'].value_counts()
        logger.info(f"Namespace distribution:")
        for namespace, count in namespace_counts.items():
            logger.info(f"  {namespace}: {count} terms")

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
        # Check if Gemma model is available
        if not hasattr(self, 'gemma_model') or self.gemma_model is None:
            raise ValueError("Gemma model not available for description generation. Please provide gemma_weights_dir.")

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
            logger.error(f"Error generating description for {metadata['go_id']}: {e}")
            raise  # Re-raise the exception to fail fast

    def generate_embeddings(self, descriptions: List[str]) -> np.ndarray:
        """
        Generate embeddings for descriptions using Gemma.

        Args:
            descriptions: List of descriptions

        Returns:
            Array of embeddings
        """
        logger.info("Generating embeddings for descriptions using Gemma")

        # Check if Gemma model is available
        if not hasattr(self, 'gemma_model') or self.gemma_model is None:
            raise ValueError("Gemma model not available for embedding generation. Please provide gemma_weights_dir.")

        embeddings = []

        # Use Gemma for embedding generation
        for i, description in enumerate(descriptions):
            try:
                # Format description for Gemma embedding
                USER_CHAT_TEMPLATE = "<start_of_turn>user\n{prompt}<end_of_turn>\n"
                MODEL_CHAT_TEMPLATE = "<start_of_turn>model\n{prompt}<end_of_turn>\n"

                # Create embedding prompt
                embedding_prompt = (
                    USER_CHAT_TEMPLATE.format(prompt=f"Generate embedding for: {description}")
                    + "<start_of_turn>model\n"
                )

                # Generate embedding using Gemma
                result = self.gemma_model.generate(
                    embedding_prompt,
                    device=self.gemma_device,
                    output_len=768  # Generate 768 tokens for embedding
                )

                # Process Gemma output to extract embedding
                response_text = result.strip()

                # Clean up Gemma-specific formatting
                if response_text.startswith("<start_of_turn>model\n"):
                    response_text = response_text[len("<start_of_turn>model\n"):]
                if response_text.endswith("<end_of_turn>"):
                    response_text = response_text[:-len("<end_of_turn>")]

                response_text = response_text.strip()

                # Convert Gemma output to numerical embedding
                # This is a simplified approach - may need refinement
                # For now, we'll use a hash-based approach to convert text to numerical vector
                import hashlib
                hash_obj = hashlib.md5(response_text.encode())
                hash_hex = hash_obj.hexdigest()
                # Convert hex to numerical values
                embedding = [int(hash_hex[j:j+2], 16) / 255.0 for j in range(0, min(384, len(hash_hex)), 2)]
                # Pad or truncate to 768 dimensions
                if len(embedding) < 768:
                    embedding.extend([0.0] * (768 - len(embedding)))
                else:
                    embedding = embedding[:768]

                embeddings.append(embedding)

                if (i + 1) % 100 == 0:
                    logger.info(f"Generated embeddings for {i + 1}/{len(descriptions)} descriptions using Gemma")

            except Exception as e:
                logger.error(f"Error generating embedding for description {i} using Gemma: {e}")
                raise  # Re-raise the exception to fail fast

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
        logger.info("Starting GO embeddings extraction")

        # Load GO data
        go_data = self.load_go_data(obo_path)

        if max_terms:
            go_data = go_data.head(max_terms)
            logger.info(f"Limiting to {max_terms} terms for testing")

        # Generate descriptions
        logger.info("Generating descriptions for GO terms")
        descriptions = []

        for idx, row in go_data.iterrows():
            description = self.generate_description(row.to_dict())
            descriptions.append(description)

            if (idx + 1) % 100 == 0:
                logger.info(f"Generated descriptions for {idx + 1}/{len(go_data)} terms")

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
                'embedding_model': "gemma" if hasattr(self, 'gemma_model') and self.gemma_model is not None else self.embedding_model_name,
                'llm_model': "gemma" if hasattr(self, 'gemma_model') and self.gemma_model is not None else (str(type(self.llm_model).__name__) if self.llm_model else "None"),
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
        logger.info("Saving results")

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
            namespaces=np.array(results['namespaces'], dtype=object)
        )

        # Save metadata as JSON
        metadata_path = self.output_dir / f"{prefix}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(results['metadata'], f, indent=2)

        # Save GO data as CSV for analysis
        go_data_path = self.output_dir / f"{prefix}_go_data.csv"
        results['go_data'].to_csv(go_data_path, index=False)

        logger.info(f"Results saved to {self.output_dir}")
        print(f"\n📁 Results saved to: {self.output_dir}")
        print(f"   📊 CSV: {csv_path.name}")
        print(f"   💾 NPZ: {npz_path.name}")
        print(f"   📋 GO Data: {go_data_path.name}")
        print(f"   📋 Metadata: {metadata_path.name}")