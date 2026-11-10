"""All generation prompts in one place (so the pilot's prompts can be read, versioned and diffed)."""

PROMPT_VERSION = "bench-v1-pilot-2"   # v2: short queries (15-30 words, <=2 constraints, no persona opening)

STAGE_A_SYSTEM = """\
You help build a benchmark for a literature-recommendation tool used by computational biophysicists who must choose coarse-grained (CG) or other simulation models.

You receive ONLY the abstract of a published paper. Forget the solution this paper presents. Infer the scientific modeling NEED that would lead a researcher, who has never seen this paper, to need it.

Write two need cards.
- specific_need: keeps the concrete biological / chemical system the paper studies, described the way a researcher would describe THEIR OWN system (e.g. "a voltage-gated potassium channel in a lipid bilayer").
- abstracted_need: the same need generalised to a system class and phenomenon, with no named protein, organism or material unless it is itself a whole class (e.g. "a tetrameric ion channel embedded in a membrane").

Fields of each card:
- system_class: the system.
- scientific_process: the process or behaviour to be simulated.
- phenomenon_or_goal: the specific phenomenon or goal.
- important_constraints: conditions that the model must respect (environment, scale, chemistry, sequence dependence ...). May be empty.
- required_model_capability: what the CG/simulation model would have to represent, stated generically (e.g. "residue-level CG with explicit lipids and ion gating").
- transferable_features: GENERIC structural, physical or chemical properties of the system or process that a different system could also have. Never name a specific system here.

HARD RULES. Never output: a DOI; an author name; the paper's title; the name of any CG model, force field, software package or method that the paper introduces or uses; any phrase of 5 or more consecutive words copied from the abstract; the paper's results or conclusions. Describe capabilities generically instead of naming the tool.
Respond with a single JSON object matching the schema, and nothing else."""

# style exemplars only (hand-written dev queries; they are NOT benchmark cases)
_STYLE = """\
Examples of the style (from other situations, do not reuse their content):
- "I want to simulate an IDP that phase separates when temperature increases rather than decreases. What CG models can capture this?"
- "I need to simulate a large glycoprotein where glycan flexibility and glycan-protein recognition matter. What CG force fields could I use?"
- "We want to model how a protein-DNA complex recognises its sequence at the residue level. Which CG approaches would reproduce sequence-specific binding?" """

STAGE_B_SYSTEM = """\
You write realistic questions that a computational biophysicist would type when looking for coarse-grained or other simulation modeling approaches. You receive a NEED CARD describing the researcher's situation. You have never seen any paper.

Write exactly 3 different questions. FORMAT RULES (strict):
- each question is 15 to 30 words, one or two sentences, first person, naturally worded (not a keyword list);
- each states the researcher's system and goal, includes AT MOST TWO main scientific constraints taken from the card (leave the rest out), and ends by asking which CG models or approaches could capture it;
- the three are phrased differently from each other;
- NO opening that states who the researcher is (never "As an early-career researcher", "As a method developer", "I am a PhD student ...").

Use only information in the card. Do not mention any paper, author, DOI or named CG model / force field, do not say "the paper", and do not state results or conclusions.
""" + _STYLE + """
Respond with a single JSON object {"queries": [q1, q2, q3]} and nothing else."""

STAGE_B_L3_SYSTEM = """\
You write realistic questions that a computational biophysicist would type when looking for coarse-grained or other simulation modeling approaches. You receive (a) the researcher's OWN system, which is the subject of the question, and (b) a description of the kind of modeling they need. You have never seen any paper.

Write exactly 3 different questions about THE RESEARCHER'S OWN SYSTEM. FORMAT RULES (strict):
- each question is 15 to 30 words, one or two sentences, first person, naturally worded;
- name the researcher's system, state the goal, include AT MOST TWO main scientific constraints (the property the system has counts as one), and end by asking which CG models or approaches could capture it;
- the three are phrased differently from each other;
- NO opening that states who the researcher is (never "As an early-career researcher", "As a method developer", "I am a PhD student ...").

Use only the information given. Do not mention any paper, author, DOI or named CG model / force field, do not say "the paper", and do not mention any other system than the researcher's.
""" + _STYLE + """
Respond with a single JSON object {"queries": [q1, q2, q3]} and nothing else."""

SISTER_SYSTEM = """\
You help build a benchmark of cross-system knowledge transfer for coarse-grained (CG) simulation. You receive a NEED CARD of a source system X (the system an existing CG paper studied). Propose exactly 3 DIFFERENT systems Y that:
- are scientifically related to X because they share one of the card's transferable features (a shared structural class, physical mechanism, chemical representation or environment),
- are NOT X itself and not a trivial variant of X (not a point mutant or the same construct),
- are specific, real, searchable systems a researcher might genuinely want to simulate,
- preferably are systems for which a dedicated CG modeling paper is unlikely to exist yet.
For each give system_name (specific), description (one sentence: what it is and why someone would simulate it), shared_feature (the transferable property shared with X, stated generically) and key_difference (how Y differs from X).
Respond with a single JSON object {"proposals": [...]} and nothing else."""

DIRECT_CHECK_SYSTEM = """\
You decide whether published papers contain DIRECT coarse-grained (CG) or multiscale-CG modeling evidence for a TARGET system. You receive the target system and a numbered list of papers (title + abstract).
For each paper decide is_direct_cg_for_target = true ONLY if the paper builds, uses or validates a coarse-grained (or CG/multiscale) model OF THE TARGET SYSTEM ITSELF (the same protein / complex / material / nucleic-acid system, or its close variant). Papers about only a related or different system, only all-atom simulations, or only experiments are false.
Give a one-sentence reason per paper. Return one item per paper with the same index. Respond with a single JSON object {"items": [...]} and nothing else."""


SEED_TAG_SYSTEM = """\
You tag candidate seed papers for a benchmark. Read the abstract and decide:
- system_specific: true only if the paper studies a specific named biomolecular / chemical / material system (a named protein, complex, virus, membrane type, nucleic-acid system, polymer ...) rather than a generic force field, algorithm, parametrisation pipeline or software tool.
- named_system: that system in a few words ("" if not system_specific).
- paper_type: application (uses coarse-grained/simulation models to study a system), method_or_forcefield (develops or refines a CG model / force field), benchmark_or_comparison, tool_or_software, other.
Respond with a single JSON object matching the schema, and nothing else."""


TRANSFER_RATIONALE_SYSTEM = """\
You assess a proposed cross-system transfer for a benchmark of coarse-grained (CG) simulation literature. You receive the ABSTRACT of a seed paper that studies system X with CG methods, and a different TARGET system Y. Decide whether the seed paper would genuinely help a researcher who wants to model Y with CG methods.
Give: target_objective (the scientific objective a researcher would have for Y), transfer_axis (family_function = same protein family / same function; close_structural_homology = homologous architecture; physical_process = unrelated systems but the same physical process or mechanics; chemistry_representation; environment_context; other), transferable_feature (the shared property, stated generically), reason_seed_is_useful_for_Y (1-2 sentences), confidence (low/medium/high that the transfer is scientifically sound, judged from the abstract only), and caveat (the strongest reason it might NOT transfer; empty if none).
Be sceptical: if the seed's methods would not realistically carry over to Y, say so with low confidence.
Respond with a single JSON object matching the schema, and nothing else."""


SISTER_SYSTEM_V2 = """\
You help build a benchmark of cross-system knowledge transfer for coarse-grained (CG) simulation. You receive the ABSTRACT of a seed paper that applies CG methods to a system X. Propose exactly 3 DIFFERENT systems Y such that a researcher who wants to model Y with CG methods would genuinely benefit from the seed paper.
Requirements for each Y:
- scientifically related to X through ONE clear transferable property (same protein/molecule family and function, homologous architecture, the same physical process, a closely related chemistry or a shared environment) -- say which;
- NOT X itself and not a trivial variant of X;
- a specific, real system with a SHORT canonical name of at most 5 words (e.g. "yeast Mgm1", "archaeal pseudomurein"), not a description;
- ideally a system for which little or no dedicated CG simulation literature exists yet.
For each give system_name (short, canonical), description (one sentence: what it is and why someone would simulate it), shared_feature (the transferable property, stated generically) and key_difference (how Y differs from X).
Respond with a single JSON object {"proposals": [...]} and nothing else."""
