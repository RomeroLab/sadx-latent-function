
## Creating alphafold structure

`SadA_Alphafold_NSLeu_AKG_ZN.pdb`. 

* Started with [Jared's `.pse` file](https://drive.google.com/file/d/1lSpJsc4x5meYRSM5tFrGf_bxntr-hUY_/view?usp=drive_link) 
* Then combined the `SadA_Alphafold` structure with NSLeu from a different structure. The exact commands used were
  ```
    select (SadA_AlphaFold) or (SadA_NSLeu_Corrected_3701_best_structure_0044 and chain X)
    save SadA_Alphafold_NSLeu_AKG_ZN.pdb, sele
  ```


