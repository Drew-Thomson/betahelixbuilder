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

The modules in `src/` implement input parsing and validation (`input.rs`),
Crick-style backbone geometry (`geometry.rs`), oligomeric bundle assembly
(`bundle.rs`), PDB output (`pdb.rs`), side-chain packing via
[dASPR](https://github.com/Drew-Thomson/dASPR) (`sidechain.rs`), and native
BUDE force-field evaluation (`energy.rs`). `optimization.rs` remains a
placeholder for future optimisation methods.

## Acknowledgments

- **[BUDEFF](https://github.com/isambard-uob/budeff)** (MIT License,
  Copyright (c) 2018 ISAMBARD / Sessions group) and the D-amino acid
  `bude_2016v1.json` parameterisation for the native Rust energy evaluator in
  `src/energy.rs`.
- **[dASPR](https://github.com/Drew-Thomson/dASPR)** (fork of FASPR) for
  mixed L/D side-chain packing.
