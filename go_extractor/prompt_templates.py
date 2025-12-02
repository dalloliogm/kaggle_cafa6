"""
Prompt templates for GO term processing.
"""

# Prompt templates for different aspects
prompt_template = {
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