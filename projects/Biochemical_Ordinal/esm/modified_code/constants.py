""" Some useful constants used throughout codebase """

DATASETS = {"avgfp": {"ds_name": "avgfp",
                      "ds_dir": "data/dms_data/avgfp",
                      "ds_fn": "data/dms_data/avgfp/avgfp.tsv",
                      "wt_aa": "SKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK",
                      "wt_ofs": 0,
                      "pdb_fn": "1gfl_cm.pdb",
                      "rosettafy_pdb_fn": "1gfl_cm.pdb",
                      "rosetta_dms_cov_fn": "data/rosetta_data/dms_coverage/avgfp_dms_cov/avgfp_dms_cov.h5",

                      "metl-l-2m-1d_dms_cov_fn": "output/htcondor_runs/inference/local_2023-03-22_10-44-59_metl_inference/output/inference/METL-L-2M-1D-GFP_8gMPQJy4/dms_avgfp/predictions_None.npy",
                      "metl-l-2m-3d_dms_cov_fn": "output/htcondor_runs/inference/local_2023-03-22_10-44-59_metl_inference/output/inference/METL-L-2M-3D-GFP_Hr4GNHws/dms_avgfp/predictions_None.npy",

                      "metl-l-2m-1d_bc1_dms_cov_fn": "output/htcondor_runs/inference/local_2023-03-22_10-44-59_metl_inference/output/inference/METL-L-2M-1D-GFP_8gMPQJy4/dms_avgfp/predictions_-1.npy",
                      "metl-l-2m-3d_bc1_dms_cov_fn": "output/htcondor_runs/inference/local_2023-03-22_10-44-59_metl_inference/output/inference/METL-L-2M-3D-GFP_Hr4GNHws/dms_avgfp/predictions_-1.npy",

                      "metl-g-20m-1d_dms_cov_fn": "output/htcondor_runs/inference/local_2023-03-22_10-44-59_metl_inference/output/inference/METL-G-20M-1D_D72M9aEp/dms_avgfp/predictions_None.npy",
                      "metl-g-20m-3d_dms_cov_fn": "output/htcondor_runs/inference/local_2023-03-22_10-44-59_metl_inference/output/inference/METL-G-20M-3D_Nr9zCKpR/dms_avgfp/predictions_None.npy",

                      "metl-g-20m-1d_bc2_dms_cov_fn": "output/htcondor_runs/inference/local_2023-03-22_10-44-59_metl_inference/output/inference/METL-G-20M-1D_D72M9aEp/dms_avgfp/predictions_-2.npy",
                      "metl-g-20m-3d_bc2_dms_cov_fn": "output/htcondor_runs/inference/local_2023-03-22_10-44-59_metl_inference/output/inference/METL-G-20M-3D_Nr9zCKpR/dms_avgfp/predictions_-2.npy",

                      "metl-g-50m-1d_dms_cov_fn": "output/htcondor_runs/inference/local_2023-03-22_10-44-59_metl_inference/output/inference/METL-G-50M-1D_auKdzzwX/dms_avgfp/predictions_None.npy",
                      "metl-g-50m-3d_dms_cov_fn": "output/htcondor_runs/inference/local_2023-03-22_10-44-59_metl_inference/output/inference/METL-G-50M-3D_6PSAzdfv/dms_avgfp/predictions_None.npy",

                      "metl-g-50m-1d_bc2_dms_cov_fn": "output/htcondor_runs/inference/local_2023-03-22_10-44-59_metl_inference/output/inference/METL-G-50M-1D_auKdzzwX/dms_avgfp/predictions_None.npy",
                      "metl-g-50m-3d_bc2_dms_cov_fn": "output/htcondor_runs/inference/local_2023-03-22_10-44-59_metl_inference/output/inference/METL-G-50M-3D_6PSAzdfv/dms_avgfp/predictions_-2.npy",

                      "esm-8m_dms_cov_fn": "output/htcondor_runs/inference/local_2023-03-22_20-25-41_esm_inference/output/inference/esm2_t6_8M_UR50D/dms_avgfp/predictions.npy",
                      "esm-35m_dms_cov_fn": "output/htcondor_runs/inference/local_2023-03-22_20-25-41_esm_inference/output/inference/esm2_t12_35M_UR50D/dms_avgfp/predictions.npy",
                      "esm-150m_dms_cov_fn": "output/htcondor_runs/inference/local_2023-03-22_20-25-41_esm_inference/output/inference/esm2_t30_150M_UR50D/dms_avgfp/predictions.npy",
                      },

            "gb1": {"ds_name": "gb1",
                    "ds_dir": "data/dms_data/gb1",
                    "ds_fn": "data/dms_data/gb1/gb1.tsv",
                    "wt_aa": "MQYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE",
                    "wt_ofs": 0,
                    "pdb_fn": "2qmt_p.pdb",
                    "rosettafy_pdb_fn": "2qmt_p.pdb",
                    "rosetta_dms_cov_fn": "data/rosetta_data/dms_coverage/gb1_dms_cov/gb1_dms_cov.h5",

                    "metl-l-2m-1d_dms_cov_fn": "output/htcondor_runs/inference/local_2023-03-22_10-44-59_metl_inference/output/inference/METL-L-2M-1D-GB1_DMfkjVzT/dms_gb1/predictions_None.npy",
                    "metl-l-2m-3d_dms_cov_fn": "output/htcondor_runs/inference/local_2023-03-22_10-44-59_metl_inference/output/inference/METL-L-2M-3D-GB1_epegcFiH/dms_gb1/predictions_None.npy",

                    "metl-l-2m-1d_bc1_dms_cov_fn": "output/htcondor_runs/inference/local_2023-03-22_10-44-59_metl_inference/output/inference/METL-L-2M-1D-GB1_DMfkjVzT/dms_gb1/predictions_-1.npy",
                    "metl-l-2m-3d_bc1_dms_cov_fn": "output/htcondor_runs/inference/local_2023-03-22_10-44-59_metl_inference/output/inference/METL-L-2M-3D-GB1_epegcFiH/dms_gb1/predictions_-1.npy",

                    "metl-g-20m-1d_dms_cov_fn": "output/htcondor_runs/inference/local_2023-03-22_10-44-59_metl_inference/output/inference/METL-G-20M-1D_D72M9aEp/dms_gb1/predictions_None.npy",
                    "metl-g-20m-3d_dms_cov_fn": "output/htcondor_runs/inference/local_2023-03-22_10-44-59_metl_inference/output/inference/METL-G-20M-3D_Nr9zCKpR/dms_gb1/predictions_None.npy",

                    "metl-g-20m-1d_bc2_dms_cov_fn": "output/htcondor_runs/inference/local_2023-03-22_10-44-59_metl_inference/output/inference/METL-G-20M-1D_D72M9aEp/dms_gb1/predictions_-2.npy",
                    "metl-g-20m-3d_bc2_dms_cov_fn": "output/htcondor_runs/inference/local_2023-03-22_10-44-59_metl_inference/output/inference/METL-G-20M-3D_Nr9zCKpR/dms_gb1/predictions_-2.npy",

                    "metl-g-50m-1d_dms_cov_fn": "output/htcondor_runs/inference/local_2023-03-22_10-44-59_metl_inference/output/inference/METL-G-50M-1D_auKdzzwX/dms_gb1/predictions_None.npy",
                    "metl-g-50m-3d_dms_cov_fn": "output/htcondor_runs/inference/local_2023-03-22_10-44-59_metl_inference/output/inference/METL-G-50M-3D_6PSAzdfv/dms_gb1/predictions_None.npy",

                    "metl-g-50m-1d_bc2_dms_cov_fn": "output/htcondor_runs/inference/local_2023-03-22_10-44-59_metl_inference/output/inference/METL-G-50M-1D_auKdzzwX/dms_gb1/predictions_-2.npy",
                    "metl-g-50m-3d_bc2_dms_cov_fn": "output/htcondor_runs/inference/local_2023-03-22_10-44-59_metl_inference/output/inference/METL-G-50M-3D_6PSAzdfv/dms_gb1/predictions_-2.npy",

                    "esm-8m_dms_cov_fn": "output/htcondor_runs/inference/local_2023-03-22_20-25-41_esm_inference/output/inference/esm2_t6_8M_UR50D/dms_gb1/predictions.npy",
                    "esm-35m_dms_cov_fn": "output/htcondor_runs/inference/local_2023-03-22_20-25-41_esm_inference/output/inference/esm2_t12_35M_UR50D/dms_gb1/predictions.npy",
                    "esm-150m_dms_cov_fn": "output/htcondor_runs/inference/local_2023-03-22_20-25-41_esm_inference/output/inference/esm2_t30_150M_UR50D/dms_gb1/predictions.npy",
                    },

            "pab1": {"ds_name": "pab1",
                     "ds_dir": "data/dms_data/pab1",
                     "ds_fn": "data/dms_data/pab1/pab1.tsv",
                     "wt_aa": "GNIFIKNLHPDIDNKALYDTFSVFGDILSSKIATDENGKSKGFGFVHFEEEGAAKEAIDALNGMLLNGQEIYVAP",
                     "wt_ofs": 0,
                     "pdb_fn": "pab1_cm.pdb",
                     "rosettafy_pdb_fn": "pab1_cm.pdb",
                     "rosetta_dms_cov_fn": "data/rosetta_data/dms_coverage/pab1_dms_cov/pab1_dms_cov.h5",

                     "metl-l-2m-1d_dms_cov_fn": "output/htcondor_runs/inference/local_2023-03-22_10-44-59_metl_inference/output/inference/METL-L-2M-1D-Pab1_UKebCQGz/dms_pab1/predictions_None.npy",
                     "metl-l-2m-3d_dms_cov_fn": "output/htcondor_runs/inference/local_2023-03-22_10-44-59_metl_inference/output/inference/METL-L-2M-3D-Pab1_2rr8V4th/dms_pab1/predictions_None.npy",

                     "metl-l-2m-1d_bc1_dms_cov_fn": "output/htcondor_runs/inference/local_2023-03-22_10-44-59_metl_inference/output/inference/METL-L-2M-1D-Pab1_UKebCQGz/dms_pab1/predictions_-1.npy",
                     "metl-l-2m-3d_bc1_dms_cov_fn": "output/htcondor_runs/inference/local_2023-03-22_10-44-59_metl_inference/output/inference/METL-L-2M-3D-Pab1_2rr8V4th/dms_pab1/predictions_-1.npy",

                     "metl-g-20m-1d_dms_cov_fn": "output/htcondor_runs/inference/local_2023-03-22_10-44-59_metl_inference/output/inference/METL-G-20M-1D_D72M9aEp/dms_pab1/predictions_None.npy",
                     "metl-g-20m-3d_dms_cov_fn": "output/htcondor_runs/inference/local_2023-03-22_10-44-59_metl_inference/output/inference/METL-G-20M-3D_Nr9zCKpR/dms_pab1/predictions_None.npy",

                     "metl-g-20m-1d_bc2_dms_cov_fn": "output/htcondor_runs/inference/local_2023-03-22_10-44-59_metl_inference/output/inference/METL-G-20M-1D_D72M9aEp/dms_pab1/predictions_-2.npy",
                     "metl-g-20m-3d_bc2_dms_cov_fn": "output/htcondor_runs/inference/local_2023-03-22_10-44-59_metl_inference/output/inference/METL-G-20M-3D_Nr9zCKpR/dms_pab1/predictions_-2.npy",

                     "metl-g-50m-1d_dms_cov_fn": "output/htcondor_runs/inference/local_2023-03-22_10-44-59_metl_inference/output/inference/METL-G-50M-1D_auKdzzwX/dms_pab1/predictions_None.npy",
                     "metl-g-50m-3d_dms_cov_fn": "output/htcondor_runs/inference/local_2023-03-22_10-44-59_metl_inference/output/inference/METL-G-50M-3D_6PSAzdfv/dms_pab1/predictions_None.npy",

                     "metl-g-50m-1d_bc2_dms_cov_fn": "output/htcondor_runs/inference/local_2023-03-22_10-44-59_metl_inference/output/inference/METL-G-50M-1D_auKdzzwX/dms_pab1/predictions_-2.npy",
                     "metl-g-50m-3d_bc2_dms_cov_fn": "output/htcondor_runs/inference/local_2023-03-22_10-44-59_metl_inference/output/inference/METL-G-50M-3D_6PSAzdfv/dms_pab1/predictions_-2.npy",

                     "esm-8m_dms_cov_fn": "output/htcondor_runs/inference/local_2023-03-22_20-25-41_esm_inference/output/inference/esm2_t6_8M_UR50D/dms_pab1/predictions.npy",
                     "esm-35m_dms_cov_fn": "output/htcondor_runs/inference/local_2023-03-22_20-25-41_esm_inference/output/inference/esm2_t12_35M_UR50D/dms_pab1/predictions.npy",
                     "esm-150m_dms_cov_fn": "output/htcondor_runs/inference/local_2023-03-22_20-25-41_esm_inference/output/inference/esm2_t30_150M_UR50D/dms_pab1/predictions.npy",
                     },

            "ube4b": {"ds_name": "ube4b",
                      "ds_dir": "data/dms_data/ube4b",
                      "ds_fn": "data/dms_data/ube4b/ube4b.tsv",
                      "wt_aa": "IEKFKLLAEKVEEIVAKNARAEIDYSDAPDEFRDPLMDTLMTDPVRLP"
                               "SGTVMDRSIILRHLLNSPTDPFNRQMLTESMLEPVPELKEQIQAWMREKQSSDH",
                      "wt_ofs": 0,
                      "pdb_fn": "ube4b_cm.pdb",
                      "rosettafy_pdb_fn": "ube4b_cm.pdb",
                      "rosetta_dms_cov_fn": "data/rosetta_data/dms_coverage/ube4b_dms_cov/ube4b_dms_cov.h5",

                      "metl-l-2m-1d_dms_cov_fn": "output/htcondor_runs/inference/local_2023-03-22_10-44-59_metl_inference/output/inference/METL-L-2M-1D-Ube4b_HscFFkAb/dms_ube4b/predictions_None.npy",
                      "metl-l-2m-3d_dms_cov_fn": "output/htcondor_runs/inference/local_2023-03-22_10-44-59_metl_inference/output/inference/METL-L-2M-3D-Ube4b_H48oiNZN/dms_ube4b/predictions_None.npy",

                      "metl-l-2m-1d_bc1_dms_cov_fn": "output/htcondor_runs/inference/local_2023-03-22_10-44-59_metl_inference/output/inference/METL-L-2M-1D-Ube4b_HscFFkAb/dms_ube4b/predictions_-1.npy",
                      "metl-l-2m-3d_bc1_dms_cov_fn": "output/htcondor_runs/inference/local_2023-03-22_10-44-59_metl_inference/output/inference/METL-L-2M-3D-Ube4b_H48oiNZN/dms_ube4b/predictions_-1.npy",

                      "metl-g-20m-1d_dms_cov_fn": "output/htcondor_runs/inference/local_2023-03-22_10-44-59_metl_inference/output/inference/METL-G-20M-1D_D72M9aEp/dms_ube4b/predictions_None.npy",
                      "metl-g-20m-3d_dms_cov_fn": "output/htcondor_runs/inference/local_2023-03-22_10-44-59_metl_inference/output/inference/METL-G-20M-3D_Nr9zCKpR/dms_ube4b/predictions_None.npy",

                      "metl-g-20m-1d_bc2_dms_cov_fn": "output/htcondor_runs/inference/local_2023-03-22_10-44-59_metl_inference/output/inference/METL-G-20M-1D_D72M9aEp/dms_ube4b/predictions_-2.npy",
                      "metl-g-20m-3d_bc2_dms_cov_fn": "output/htcondor_runs/inference/local_2023-03-22_10-44-59_metl_inference/output/inference/METL-G-20M-3D_Nr9zCKpR/dms_ube4b/predictions_-2.npy",

                      "metl-g-50m-1d_dms_cov_fn": "output/htcondor_runs/inference/local_2023-03-22_10-44-59_metl_inference/output/inference/METL-G-50M-1D_auKdzzwX/dms_ube4b/predictions_None.npy",
                      "metl-g-50m-3d_dms_cov_fn": "output/htcondor_runs/inference/local_2023-03-22_10-44-59_metl_inference/output/inference/METL-G-50M-3D_6PSAzdfv/dms_ube4b/predictions_None.npy",

                      "metl-g-50m-1d_bc2_dms_cov_fn": "output/htcondor_runs/inference/local_2023-03-22_10-44-59_metl_inference/output/inference/METL-G-50M-1D_auKdzzwX/dms_ube4b/predictions_-2.npy",
                      "metl-g-50m-3d_bc2_dms_cov_fn": "output/htcondor_runs/inference/local_2023-03-22_10-44-59_metl_inference/output/inference/METL-G-50M-3D_6PSAzdfv/dms_ube4b/predictions_-2.npy",

                      "esm-8m_dms_cov_fn": "output/htcondor_runs/inference/local_2023-03-22_20-25-41_esm_inference/output/inference/esm2_t6_8M_UR50D/dms_ube4b/predictions.npy",
                      "esm-35m_dms_cov_fn": "output/htcondor_runs/inference/local_2023-03-22_20-25-41_esm_inference/output/inference/esm2_t12_35M_UR50D/dms_ube4b/predictions.npy",
                      "esm-150m_dms_cov_fn": "output/htcondor_runs/inference/local_2023-03-22_20-25-41_esm_inference/output/inference/esm2_t30_150M_UR50D/dms_ube4b/predictions.npy",
                      },

            "dlg4": {"ds_name": "dlg4",
                     "ds_dir": "data/dms_data/dlg4",
                     "ds_fn": "data/dms_data/dlg4/dlg4.tsv",
                     "wt_aa": "GSTGLGFNIVGGEDGEGIFISFILAGGPADLSGELRKGDQILSVNGVDLRNASHEQAAIALKNAGQ",
                     "wt_ofs": 0,
                     "pdb_fn": "6qji_p_trunc.pdb",
                     # because DLG4 uses a longer PDB sequence for Rosetta runs, need to offset our DMS sequence
                     # when creating a Rosetta run and converting from DMS indexing to rosettafy PDB indexing
                     # this is the offset to go from DMS --> Rosettafy
                     # to go from Rosettafy --> DMS, multiply by -1
                     "rosettafy_pdb_offset": 9,
                     "rosettafy_pdb_fn": "6qji_p.pdb",
                     "rosetta_dms_cov_fn": "data/rosetta_data/dms_coverage/dlg4_dms_cov/dlg4_dms_cov.h5",

                     "metl-l-2m-1d_dms_cov_fn": "output/htcondor_runs/inference/local_2023-03-22_10-44-59_metl_inference/output/inference/METL-L-2M-1D-DLG4_4aoa5FiW/dms_dlg4/predictions_None.npy",
                     "metl-l-2m-3d_dms_cov_fn": "output/htcondor_runs/inference/local_2023-03-22_10-44-59_metl_inference/output/inference/METL-L-2M-3D-DLG4_638NCaiM/dms_dlg4/predictions_None.npy",

                     "metl-l-2m-1d_bc1_dms_cov_fn": "output/htcondor_runs/inference/local_2023-03-22_10-44-59_metl_inference/output/inference/METL-L-2M-1D-DLG4_4aoa5FiW/dms_dlg4/predictions_-1.npy",
                     "metl-l-2m-3d_bc1_dms_cov_fn": "output/htcondor_runs/inference/local_2023-03-22_10-44-59_metl_inference/output/inference/METL-L-2M-3D-DLG4_638NCaiM/dms_dlg4/predictions_-1.npy",

                     "metl-g-20m-1d_dms_cov_fn": "output/htcondor_runs/inference/local_2023-03-22_10-44-59_metl_inference/output/inference/METL-G-20M-1D_D72M9aEp/dms_dlg4/predictions_None.npy",
                     "metl-g-20m-3d_dms_cov_fn": "output/htcondor_runs/inference/local_2023-03-22_10-44-59_metl_inference/output/inference/METL-G-20M-3D_Nr9zCKpR/dms_dlg4/predictions_None.npy",

                     "metl-g-20m-1d_bc2_dms_cov_fn": "output/htcondor_runs/inference/local_2023-03-22_10-44-59_metl_inference/output/inference/METL-G-20M-1D_D72M9aEp/dms_dlg4/predictions_-2.npy",
                     "metl-g-20m-3d_bc2_dms_cov_fn": "output/htcondor_runs/inference/local_2023-03-22_10-44-59_metl_inference/output/inference/METL-G-20M-3D_Nr9zCKpR/dms_dlg4/predictions_-2.npy",

                     "metl-g-50m-1d_dms_cov_fn": "output/htcondor_runs/inference/local_2023-03-22_10-44-59_metl_inference/output/inference/METL-G-50M-1D_auKdzzwX/dms_dlg4/predictions_None.npy",
                     "metl-g-50m-3d_dms_cov_fn": "output/htcondor_runs/inference/local_2023-03-22_10-44-59_metl_inference/output/inference/METL-G-50M-3D_6PSAzdfv/dms_dlg4/predictions_None.npy",

                     "metl-g-50m-1d_bc2_dms_cov_fn": "output/htcondor_runs/inference/local_2023-03-22_10-44-59_metl_inference/output/inference/METL-G-50M-1D_auKdzzwX/dms_dlg4/predictions_-2.npy",
                     "metl-g-50m-3d_bc2_dms_cov_fn": "output/htcondor_runs/inference/local_2023-03-22_10-44-59_metl_inference/output/inference/METL-G-50M-3D_6PSAzdfv/dms_dlg4/predictions_-2.npy",

                     "esm-8m_dms_cov_fn": "output/htcondor_runs/inference/local_2023-03-22_20-25-41_esm_inference/output/inference/esm2_t6_8M_UR50D/dms_dlg4/predictions.npy",
                     "esm-35m_dms_cov_fn": "output/htcondor_runs/inference/local_2023-03-22_20-25-41_esm_inference/output/inference/esm2_t12_35M_UR50D/dms_dlg4/predictions.npy",
                     "esm-150m_dms_cov_fn": "output/htcondor_runs/inference/local_2023-03-22_20-25-41_esm_inference/output/inference/esm2_t30_150M_UR50D/dms_dlg4/predictions.npy",
                     }
            }

# list of chars that can be encountered in any sequence
CHARS = ["*", "A", "C", "D", "E", "F", "G", "H", "I", "K", "L",
         "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y"]

# number of chars
NUM_CHARS = 21  # len(CHARS)

# dictionary mapping chars->int
C2I_MAPPING = {c: i for i, c in enumerate(CHARS)}

C2I_MAPPING_2 = {
    "*": 0,
    "A": 1,
    "C": 2,
    "D": 3,
    "E": 4,
    "F": 5,
    "G": 6,
    "H": 7,
    "I": 8,
    "K": 9,
    "L": 10,
    "M": 11,
    "N": 12,
    "P": 13,
    "Q": 14,
    "R": 15,
    "S": 16,
    "T": 17,
    "V": 18,
    "W": 19,
    "Y": 20,
    "CLS": 21,
    "PAD": 22
}

AA_3TO1_MAP = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
               'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N',
               'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W',
               'ALA': 'A', 'VAL': 'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M',
               'TER': '*'}

ROSETTA_ATTRIBUTES = ('total_score', 'dslf_fa13', 'fa_atr', 'fa_dun', 'fa_elec',
                      'fa_intra_rep', 'fa_intra_sol_xover4', 'fa_rep', 'fa_sol',
                      'hbond_bb_sc', 'hbond_lr_bb', 'hbond_sc', 'hbond_sr_bb', 'lk_ball_wtd',
                      'omega', 'p_aa_pp', 'pro_close', 'rama_prepro', 'ref', 'yhh_planarity',
                      'filter_total_score', 'buried_all', 'buried_np', 'contact_all',
                      'contact_buried_core', 'contact_buried_core_boundary', 'degree',
                      'degree_core', 'degree_core_boundary', 'exposed_hydrophobics',
                      'exposed_np_AFIMLWVY', 'exposed_polars', 'exposed_total',
                      'one_core_each', 'pack', 'res_count_all', 'res_count_buried_core',
                      'res_count_buried_core_boundary', 'res_count_buried_np_core',
                      'res_count_buried_np_core_boundary', 'ss_contributes_core', 'ss_mis',
                      'total_hydrophobic', 'total_hydrophobic_AFILMVWY', 'total_sasa',
                      'two_core_each', 'unsat_hbond', 'centroid_total_score', 'cbeta',
                      'cenpack', 'env', 'hs_pair', 'linear_chainbreak', 'overlap_chainbreak',
                      'pair', 'rg', 'rsigma', 'sheet', 'ss_pair', 'vdw')
