# betahelixbuilder

A planned Rust tool for parametric modelling of beta-helix peptides such as
gramicidin and polytheonamide. Here, beta helices are peptide helices with
alternating D- and L-amino acids, not beta solenoids.

This repository is an early scaffold: it does not yet generate structures,
provide a command-line interface, or calculate energies.

## Initial goal

Given a sequence with explicit residue stereochemistry and parameters such as
helix handedness, radius, and oligomeric state, generate a PDB structure for
an individual peptide or an oligomeric bundle. Bundle modelling should allow
the peptide backbone to flex to accommodate supercoiling, rather than treating
each helix only as a rigid body.

## Roadmap

1. **Define inputs and conventions.** Specify sequence and D/L notation,
   parameter units and defaults, supported residue types, handedness, and
   validation rules. Define how chain identities and residue numbering map to
   PDB output.
2. **Build a single beta helix.** Develop and validate a parametric backbone
   model, including stereochemistry, bond geometry, and PDB export. Compare
   representative structures and geometric invariants against reference data.
3. **Model bundles and supercoiling.** Arrange multiple helices with explicit
   bundle geometry and implement backbone flexibility along supercoiled paths.
   Validate chain placement, connectivity, and backbone geometry.
4. **Add side-chain packing and energy evaluation.** Design interfaces for
   [dASPR](https://github.com/Drew-Thomson/dASPR) side-chain packing with D
   residues and for a suitable energy force field. Define data exchange,
   supported residue conventions, and validation before integrating either.
5. **Explore optimisation and additional geometries.** Investigate genetic
   algorithms or differential evolution for structure optimisation, and
   extend the geometry model to double helices and other complex structures.

## Code layout

The files in `src/` are placeholders for input handling, helix geometry,
bundles, PDB output, side-chain packing, energy evaluation, and optimisation.
They do not yet define a stable API. The scaffold can be checked with
`cargo check` and `cargo test`.
