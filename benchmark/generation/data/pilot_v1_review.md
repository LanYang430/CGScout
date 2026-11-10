# BEAM-bench v1 pilot -- pick one question per case (judge NATURALNESS, not fit to the paper)

## Q01  L1  (membrane)
- need: A TRPC3 ion channel embedded in a neuronal lipid membrane | Lipid-dependent regulation of channel conformational dynamics and ion permeation
1. I want to simulate TRPC3 in a neuronal lipid membrane to determine how polyphosphorylated lipid binding regulates activity through pore gating. Which CG models or approaches could capture it?
2. I need to model TRPC3 channel dynamics and ion permeation with residue-level lipid-site interactions and explicit anionic phosphoinositides. Which CG models or approaches could capture this?
3. How can I simulate TRPC3’s transmembrane and pore architecture to connect lipid binding with multistep allosteric opening and closing? Which CG models or approaches could capture it?

## Q02  L1  (membrane)
- need: An Escherichia coli outer membrane containing lipopolysaccharides and native outer-membrane proteins, exposed to polymyxin B1 | Association of an antimicrobial peptide with, organization within, and possible translocation across the bacterial outer membrane
1. I want to model a Gram-negative bacterial outer membrane with lipopolysaccharides and native proteins to track polymyxin B1 binding and translocation. Which CG models or approaches could capture it?  **[flags: copied_phrase:'gram negative bacterial outer', copied_phrase:'negative bacterial outer membrane']**
2. I need to simulate polymyxin B1 associating with an asymmetric bacterial outer membrane, emphasizing electrostatic peptide–lipopolysaccharide interactions and lipid–protein reorganization. Which CG models or approaches could capture it?
3. I want to examine polymyxin B1 aggregation and movement through a dense bacterial outer-membrane barrier while preserving lateral connectivity. Which CG models or approaches could capture it?

## Q03  L1  (idp_condensate)
- need: eIF4B, a eukaryotic translation-initiation factor containing a large intrinsically disordered region and a folded RNA-binding domain | Conformational rearrangement, self-association, oligomer formation, and eventual condensed-phase separation of the disordered region
1. I want to model eIF4B as it progresses from monomers through oligomers to condensates, capturing its disordered-region sequence and variable ionic strength. Which CG models or approaches could capture it?
2. I need to simulate eIF4B’s folded RNA-binding domain and disordered region during self-association, while representing macromolecular crowding. Which CG models or approaches could capture it?
3. I want to resolve how eIF4B’s disordered region forms liquid-like condensates from reversible oligomers while allowing covalent modification or partner binding. Which CG models or approaches could capture it?

## Q04  L1  (molecular_machine)
- need: the bacterial 30S ribosomal subunit composed of 16S ribosomal RNA and multiple ribosomal proteins | stepwise assembly of a protein–RNA biomolecular complex following an experimentally established assembly sequence
1. I want to model the 30S ribosomal subunit assembling through ordered protein-binding stages and determine how 16S rRNA conformation and flexibility change. Which coarse-grained models or approaches could capture it?  **[flags: title_ngram:'the 30s ribosomal subunit']**
2. I need to simulate the bacterial 30S subunit to determine how protein binding reshapes sequence-dependent 16S rRNA tertiary structure and collective motions. Which coarse-grained models or approaches could capture it?
3. I want to represent formation of the 30S ribosomal subunit in aqueous, ion-stabilized conditions while tracking successive protein associations and RNA flexibility. Which coarse-grained models or approaches could capture it?  **[flags: title_ngram:'of the 30s ribosomal subunit']**

## Q05  L1  (cytoskeleton)
- need: Actin protein filaments with and without bound actin-severing regulatory proteins | Structural dynamics and mechanical deformation of cytoskeletal filaments, including how regulatory-protein binding may alter filament behavior relevant to severing
1. I want to compare flexibility and stiffness in bare and actin-severing-protein-bound filaments while preserving realistic subunit organization. Which CG models could capture this?
2. How can I model regulatory-protein binding at actin subunit interfaces and assess its effects on long-range filament deformation? Which CG approaches could capture it?
3. I need to extract low-frequency collective motions and continuum-scale bending, stretching, and torsional properties for actin filaments with or without regulators. Which CG models could capture this?

## Q06  L1  (rna)
- need: An adenine-responsive add A-riboswitch RNA | Ligand-dependent folding and unfolding of the RNA scaffold as adenine binds or dissociates
1. I want to simulate an adenine-responsive add A-riboswitch RNA to track fluctuations and unfolding after adenine dissociation while preserving native contacts. Which CG models or approaches could capture it?
2. I need to model adenine-bound and adenine-free states of an add A-riboswitch RNA, keeping ligand pockets and structural elements distinguishable during unfolding. Which CG models or approaches could capture it?
3. I want to predict how adenine occupancy stabilizes distal RNA regions and orders progressive contact loss in the add A-riboswitch. Which CG models or approaches could capture this?

## Q07  L2  (membrane)
- need: A membrane-associated glycosyltransferase that modifies lipid-anchored carbohydrate substrates | Membrane-coupled enzymatic catalysis involving substrate recognition, active-site rearrangement, and transient penetration of protein segments into a bilayer
1. I want to model a membrane-associated glycosyltransferase accessing lipid-anchored carbohydrate substrates while flexible surface regions transiently penetrate the bilayer. Which CG models or approaches could capture it?
2. I need to explain how a lipid-facing enzyme’s membrane-interacting sequence elements regulate catalysis, comparing sequence variants through protein, lipid, and carbohydrate motions. Which CG models or approaches could capture it?
3. I want to simulate membrane-coupled catalysis in a glycosyltransferase, focusing on active-site rearrangement when membrane anchoring and bilayer packing restrict carbohydrate-substrate access. Which CG models or approaches could capture it?

## Q08  L2  (idp_condensate)
- need: Multicomponent condensates made from multivalent biomacromolecules | Phase separation, internal structural organization, and relaxation dynamics of biomolecular condensates
1. I want to simulate multicomponent biomolecular condensates to relate interaction patterns and spatial heterogeneity to phase behavior and relaxation dynamics. Which CG models or approaches could capture it?
2. I need to model a dense aqueous condensate with multivalent domains of distinct affinities, connecting molecular interactions to emergent structure and relaxation. Which CG models or approaches could capture it?
3. I am studying multicomponent biomolecular condensates and want to resolve network formation across molecular and mesoscopic scales while distinguishing interaction classes. Which CG models or approaches could capture it?

## Q09  L2  (idp_condensate)
- need: A multicomponent mixture of intrinsically disordered regulatory proteins and a recruited macromolecular client | Sequence-dependent multicomponent liquid–liquid phase separation and client co-condensation
1. I want to model a mixture of intrinsically disordered regulatory proteins and a recruited macromolecular client to predict condensate formation and client recruitment. Which CG models or approaches could capture it?  **[flags: format_length:31_words]**
2. I need to simulate sequence-dependent phase separation among disordered regulatory proteins and a recruited client, including competing association and condensate composition. Which CG models or approaches could capture this?
3. I aim to explain selective client co-condensation in a multicomponent mixture of intrinsically disordered regulatory proteins across homogeneous and phase-separated concentration regimes. Which CG models or approaches could capture this?

## Q10  L2  (chromatin_dna)
- need: A sequence-defined nucleoprotein complex in which a flexible nucleic-acid polymer is wrapped around or bound extensively to a compact protein assembly | Conformational dynamics and slow structural rearrangements of the protein–nucleic-acid assembly
1. I want to simulate a sequence-defined nucleic-acid polymer bound extensively to a compact protein assembly, tracking breathing and metastable-state exchange under physiological solvent and ionic conditions. Which CG models or approaches could capture it?  **[flags: format_length:34_words]**
2. I need to model how nucleic-acid sequence patterns influence polymer repositioning, accessibility, and exchange among slow conformational states during long simulations. Which CG models or approaches could capture it?
3. I plan to study local fluctuations and global rearrangements in a compact protein–nucleic-acid assembly while preserving polymer geometry and specific contacts. Which CG models or approaches could capture it?

## Q11  L2  (virus_capsid)
- need: Deformable anisotropic protein subunits assembling into a finite polyhedral shell | Interacting-particle self-assembly with conformational adaptation, defect correction, and closed-shell formation
1. I want to model deformable anisotropic protein subunits assembling into a polyhedral shell, comparing ordered assembly with kinetic trapping from directional contacts. Which CG models or approaches could capture it?
2. I need to simulate protein subunits that preserve a preferred shape while deforming elastically to close a finite shell. Which coarse-grained models or approaches could capture this?
3. I want to study deformable protein subunits forming a finite shell and determine how interaction strength versus elastic resistance produces defective intermediates. Which CG models or approaches could capture this?

## Q12  L2  (dna_nanotech)
- need: Sequence-programmed, multistranded nucleic-acid nanostructures with three-dimensional architectures | Self-assembly and conformational dynamics of a hierarchical molecular nanostructure in solution
1. I want to simulate sequence-programmed, multistranded nucleic-acid nanostructures in solution to determine equilibrium conformations, while retaining connectivity and complementary pairing. Which CG models or approaches could capture it?
2. I need to predict local-to-global structural and mechanical properties of large nucleic-acid assemblies, including hierarchical motifs and junctions across mesoscopic scales. Which CG models or approaches could capture it?
3. I want to model self-assembly and conformational dynamics of three-dimensional nucleic-acid architectures, efficiently sampling long-timescale local fluctuations and global reorganization. Which CG models or approaches could capture it?  **[flags: title_ngram:'and conformational dynamics of']**

## Q13  L2  (membrane)
- need: A crowded multicomponent biological membrane containing interacting membrane proteins or protein complexes | Simulating membrane organization, protein interactions, and nanoscale structural dynamics over biologically relevant scales
1. I want to simulate a multicomponent membrane with interacting proteins to predict how organization influences function, preserving heterogeneous lipids and nanoscale resolution. Which CG models or approaches could capture it?
2. I need to model interacting proteins in a heterogeneous crowded membrane and test how their spatial organization affects function across larger scales. Which CG models or approaches could capture it?
3. I want predictive simulations of a crowded biological membrane with multiple protein complexes and lipids, capturing specific interactions and emergent spatial patterning. Which CG models or approaches could capture it?

## Q14  L2  (protein_surface)
- need: A charged globular biomolecule interacting with chemically and electrostatically heterogeneous aqueous solid interfaces | Interfacial adsorption accompanied by molecular reorientation and structural response
1. I want to simulate a charged globular biomolecule adsorbing and reorienting at tunable aqueous solid interfaces, preserving charge heterogeneity and salt screening. Which CG approaches could capture this?
2. How can I model biomolecular adsorption that changes orientation and deformation across hydrophobic, hydrated, neutral, and charged surfaces while representing mobile ions? Which CG models could capture it?
3. I need microsecond-scale simulations of rare association and interfacial rearrangement for a flexible charged biomolecule without overwhelming its structure through surface binding. Which CG approaches could capture this?

## Q15  L3  (ion_channel)
- seed system X: MscS  ->  researcher's system Y: **Arabidopsis MSL10**  (verified_no_direct=True; papers checked=46, transfer axis=family_function, confidence=high)
- need: A mechanosensitive membrane-embedded ion channel | Membrane-stress-driven conformational gating
1. I want to model Arabidopsis MSL10's tension-dependent gating and predict how bilayer stress drives nonconducting-to-conducting transitions; which CG approaches could capture this?
2. How can I simulate Arabidopsis MSL10 to determine how its preserved channel architecture couples to membrane tension during switching between functional states, and which CG models could capture it?
3. I need a coarse-grained representation of Arabidopsis MSL10 that preserves topology and resolves geometry-dependent gating under lateral bilayer stress; which CG approaches could capture these transitions?

## Q16  L3  (ribonucleoprotein)
- seed system X: E. coli 70S ribosome  ->  researcher's system Y: **chloroplast 70S ribosome**  (verified_no_direct=True; papers checked=59, transfer axis=family_function, confidence=high)
- need: A large heterogeneous ribonucleoprotein complex organized into interacting subunits and dynamic domains | Simulating collective conformational dynamics and mechanical coupling among multiple molecular components
1. I want to simulate the chloroplast 70S ribosome to map intercomponent interactions controlling collective flexibility and communication, while retaining domain-scale subunit interfaces. Which CG models or approaches could capture it?
2. I need a multiscale simulation of the chloroplast 70S ribosome to determine how nucleic-acid and protein interactions drive rearrangements despite large size. Which CG models or approaches could capture it?
3. I aim to extract the effective mechanical interaction network of the chloroplast 70S ribosome and calibrate fluctuations across mobile domains. Which CG models or approaches could capture it?

## Q17  L3  (membrane_insertion)
- seed system X: YidC  ->  researcher's system Y: **chloroplast Alb3 insertase**  (verified_no_direct=True; papers checked=42, transfer axis=family_function, confidence=high)
- need: A membrane protein insertase or translocase interacting with a nascent transmembrane α-helical substrate | Assisted membrane insertion and folding of a newly synthesized membrane protein
1. I want to simulate chloroplast Alb3 insertase binding and inserting a single transmembrane protein, capturing its folding pathway in an explicit lipid environment. Which CG models or approaches could capture this?  **[flags: title_ngram:'a single transmembrane protein', format_length:31_words]**
2. I need to model how chloroplast Alb3 insertase transfers a flexible transmembrane substrate into the thylakoid membrane and selects its sequence-dependent topology. Which CG models or approaches could capture this?
3. I want to resolve the kinetics of chloroplast Alb3 insertase-assisted membrane insertion, including heterogeneous binding and coupled helix formation during native folding. Which CG models or approaches could capture this?

## Q18  L3  (motility_filament)
- seed system X: bacterial flagellum  ->  researcher's system Y: **archaeal archaellum**  (verified_no_direct=True; papers checked=38, transfer axis=physical_process, confidence=medium)
- need: A long, slender helical macromolecular filament or polymeric assembly immersed in fluid | Torque-driven mechanical response, including coupled bending, twisting, and rearrangement of repeating subunits
1. I want to model the archaeal archaellum, a rotating helical motility filament, to explain torque-driven bending-twisting transitions between global conformations; which CG approaches could capture it?
2. I need to model the rotating archaeal archaellum to predict direction-dependent conformational transitions under torque in long filaments with fluid coupling; which CG approaches could capture it?
3. I want to explain how local heterogeneity in the archaeal archaellum controls torque-driven bending and twisting of its repeating helical filament; which CG models could capture this?

## Q19  L3  (transcription)
- seed system X: RNA polymerase II transcription initiation complex  ->  researcher's system Y: **Sulfolobus RNA polymerase**  (verified_no_direct=True; papers checked=90, transfer axis=close_structural_homology, confidence=medium)
- need: A multisubunit DNA-dependent polymerase initiation complex bound to a promoter or regulatory DNA region | Localized DNA opening and rearrangement of the initiation complex before or during early polymerization
1. I want to simulate Sulfolobus RNA polymerase with TBP- and TFB-type factors to determine how promoter DNA opening evolves while preserving sequence specificity. Which CG approaches could capture it?
2. I need to model Sulfolobus RNA polymerase initiation with TBP- and TFB-type factors, asking how promoter DNA rearrangement couples duplex opening to protein gating. Which CG approaches could capture it?
3. I want a multiresolution simulation of Sulfolobus RNA polymerase, TBP- and TFB-type factors, and promoter DNA to identify metastable initiation states during local DNA opening despite the assembly's size. Which CG models could capture it?  **[flags: format_length:35_words]**

## Q20  L3  (membrane_remodeling)
- seed system X: dynamin  ->  researcher's system Y: **yeast Mgm1**  (verified_no_direct=True; papers checked=87, transfer axis=family_function, confidence=medium)
- need: A nucleotide-regulated oligomeric membrane-remodeling protein assembly surrounding a lipid tubule | Chemically triggered conformational and assembly rearrangements coupled to deformation of a flexible membrane
1. I want to model yeast Mgm1, whose nucleotide-dependent assembly constricts the inner membrane, and determine how chemical cycling drives remodeling in an explicit flexible membrane. Which CG approaches could capture this?  **[flags: format_length:31_words]**
2. For yeast Mgm1, I need to determine how nucleotide-dependent assembly of a membrane-bound oligomer generates constriction and possible topological remodeling. Which CG models could capture this?
3. I am seeking a multiscale simulation of yeast Mgm1 that links local nucleotide-dependent conformational changes to constriction of a curved, deformable inner membrane. Which CG models or approaches could capture this?  **[flags: format_length:31_words]**
