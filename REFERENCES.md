# References

Works this repository builds on or uses. Metadata of every entry was checked against the Crossref or arXiv record of the work.

## Where the published scores in the README come from

"Best published" is the strongest number printed in the source paper's results table for that benchmark, whichever row it is. These are scores of **fine-tuned** models (the CellPPD value is a baseline, CheMeleon, that the PeptideCLM-2 authors ran in their table; the PL_cCPP and PL_cSol values need ESM-2 as a second model). Recomputing the MCC from the prediction files released with PeptideCLM-2 shows that its published MCC values are at the best decision threshold chosen on the test set.

| Benchmark | Best published | Source |
|---|---|---|
| CellPPD (MCC) | 0.892 | CheMeleon row, Table 1 of Feller et al. 2026; PeptideCLM-2 itself 0.875. CheMeleon: Burns et al. 2026 |
| THPep (MCC) | 0.756 | PeptideCLM-2 (MLM), Table 1 of Feller et al. 2026 |
| PepMSND (MCC) | 0.749 | PeptideCLM-2 (MTR) with the descriptor-KAN pipeline of PepMSND (Hu et al. 2025), Table 1 of Feller et al. 2026 |
| AmpHGT (MCC) | 0.884 | PeptideCLM-2 (MLM), Table 1 of Feller et al. 2026 |
| PL_cCPP (AUROC) | 0.885 | PepLand + ESM-2, Table 1 of Zhang et al. 2025 (PepLand alone 0.838) |
| PL_cSol (AUROC) | 0.730 | PepLand + ESM-2, Table 1 of Zhang et al. 2025 (PepLand alone 0.662) |
| PL_ncCPP (Spearman) | 0.628 | PepLand, Table 1 of Zhang et al. 2025 |

## Reference list

| Work | Used for |
|---|---|
| Alain, G. & Bengio, Y. 2016. Understanding intermediate layers using linear classifier probes. arXiv preprint arXiv:1610.01644. | linear probes (frozen-embedding evaluation) |
| Arús-Pous, J., Johansson, S. V., Prykhodko, O., Bjerrum, E. J., Tyrchan, C., Reymond, J.-L., Chen, H. & Engkvist, O. 2019. Randomized SMILES strings improve the quality of molecular generative models. Journal of Cheminformatics, 11, 71. https://doi.org/10.1186/s13321-019-0393-0 | randomized SMILES (random spellings) |
| Bjerrum, E. J. 2017. SMILES enumeration as data augmentation for neural network modeling of molecules. arXiv preprint arXiv:1703.07076. | SMILES enumeration as data augmentation (random spellings) |
| Burns, J. W., Zalte, A. S., Abreu, C. R. A., Sieg, J., Feldmann, C., Mathea, M. & Green, W. H. 2026. Deep learning foundation models for low-data regimes from classical molecular descriptors. Journal of Chemical Information and Modeling, 66, 10692-10701. https://doi.org/10.1021/acs.jcim.6c01546 | CheMeleon: the best published CellPPD score |
| Chicco, D. & Jurman, G. 2020. The advantages of the Matthews correlation coefficient (MCC) over F1 score and accuracy in binary classification evaluation. BMC Genomics, 21, 6. https://doi.org/10.1186/s12864-019-6413-7 | advantages of the Matthews correlation coefficient |
| Feller, A. L., Secor, M., Swanson, S., Wilke, C. O. & Deibler, K. 2026. Scaling SMILES-based chemical language models for therapeutic peptide engineering. Journal of Chemical Information and Modeling, 66, 8110-8122. https://doi.org/10.1021/acs.jcim.6c00652 | PeptideCLM-2: input model of the teacher; source of four benchmarks and of the best published scores |
| He, Y., Song, X., Wan, H. & Zhao, X. 2025. AmpHGT: expanding prediction of antimicrobial activity in peptides containing non-canonical amino acids using multi-view constrained heterogeneous graph transformer. BMC Biology, 23, 184. https://doi.org/10.1186/s12915-025-02253-4 | AmpHGT benchmark (antimicrobial peptides) |
| Hinton, G., Vinyals, O. & Dean, J. 2015. Distilling the knowledge in a neural network. arXiv preprint arXiv:1503.02531. | knowledge distillation |
| Hu, H., Zhang, C., Xu, Z., Guo, J., Su, A., Li, C. & Duan, H. 2025. PepMSND: integrating multi-level feature engineering and comprehensive databases to enhance in vitro / in vivo peptide blood stability prediction. Digital Discovery, 4, 2478-2490. https://doi.org/10.1039/d5dd00118h | PepMSND benchmark (blood stability) |
| Kim, S., Chen, J., Cheng, T., Gindulyte, A., He, J., He, S. et al. 2023. PubChem 2023 update. Nucleic Acids Research, 51, D1373-D1380. https://doi.org/10.1093/nar/gkac956 | PubChem (source of non-canonical and cyclic peptides in the corpus) |
| Kumar, S., Solanki, S., Gupta, M., Chauhan, S., Mohanty, S. K., Satija, S. et al. 2026. Scalable molecular representations enabled by multimodal fusion and sequence distillation. Nature Communications, 17, 10981. https://doi.org/10.1038/s41467-026-77700-z | Chemical Dice Integrator: the fuse-then-distil idea this project follows |
| Kumar, V., Agrawal, P., Kumar, R., Bhalla, S., Usmani, S. S., Varshney, G. C. & Raghava, G. P. S. 2018. Prediction of cell-penetrating potential of modified peptides containing natural and chemically modified residues. Frontiers in Microbiology, 9, 725. https://doi.org/10.3389/fmicb.2018.00725 | CellPPD benchmark (cell-penetrating peptides with modifications) |
| Li, J., Yanagisawa, K., Sugita, M., Fujie, T., Ohue, M. & Akiyama, Y. 2023. CycPeptMPDB: A comprehensive database of membrane permeability of cyclic peptides. Journal of Chemical Information and Modeling, 63, 2240-2250. https://doi.org/10.1021/acs.jcim.2c01573 | CycPeptMPDB (membrane permeability of cyclic peptides) |
| Lin, Z., Akin, H., Rao, R., Hie, B., Zhu, Z., Lu, W. et al. 2023. Evolutionary-scale prediction of atomic-level protein structure with a language model. Science, 379, 1123-1130. https://doi.org/10.1126/science.ade2574 | ESM-2: reconstruction target of the teacher |
| Loshchilov, I. & Hutter, F. 2019. Decoupled weight decay regularization. International Conference on Learning Representations (ICLR). arXiv preprint arXiv:1711.05101. | AdamW |
| Matthews, B. W. 1975. Comparison of the predicted and observed secondary structure of T4 phage lysozyme. Biochimica et Biophysica Acta (BBA) - Protein Structure, 405, 442-451. https://doi.org/10.1016/0005-2795(75)90109-9 | Matthews correlation coefficient |
| Men, X., Xu, M., Zhang, Q., Wang, B., Lin, H., Lu, Y., Han, X. & Chen, W. 2024. ShortGPT: Layers in large language models are more redundant than you expect. arXiv preprint arXiv:2403.03853. | ShortGPT, Block Influence layer removal (the midterm compression of PeptideCLM-2) |
| Park, W., Kim, D., Lu, Y. & Cho, M. 2019. Relational knowledge distillation. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 3962-3971. https://doi.org/10.1109/cvpr.2019.00409 | relational knowledge distillation (the relational loss of the student) |
| Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G. et al. 2019. PyTorch: An imperative style, high-performance deep learning library. Advances in Neural Information Processing Systems (NeurIPS). arXiv preprint arXiv:1912.01703. | PyTorch |
| Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O. et al. 2011. Scikit-learn: Machine learning in Python. Journal of Machine Learning Research, 12, 2825-2830. | scikit-learn (the four learners) |
| Rogers, D. & Hahn, M. 2010. Extended-connectivity fingerprints. Journal of Chemical Information and Modeling, 50, 742-754. https://doi.org/10.1021/ci100050t | extended-connectivity fingerprints (ECFP counts): third input of the teacher |
| Sennrich, R., Haddow, B. & Birch, A. 2016. Neural machine translation of rare words with subword units. Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 1715-1725. https://doi.org/10.18653/v1/p16-1162 | byte-pair encoding (the student's tokenizer) |
| Shazeer, N. 2020. GLU variants improve transformer. arXiv preprint arXiv:2002.05202. | SwiGLU |
| Shoombuatong, W., Schaduangrat, N., Pratiwi, R. & Nantasenamat, C. 2019. THPep: A machine learning-based approach for predicting tumor homing peptides. Computational Biology and Chemistry, 80, 441-451. https://doi.org/10.1016/j.compbiolchem.2019.05.008 | THPep benchmark (tumour-homing peptides) |
| Su, J., Ahmed, M., Lu, Y., Pan, S., Bo, W. & Liu, Y. 2024. RoFormer: Enhanced transformer with rotary position embedding. Neurocomputing, 568, 127063. https://doi.org/10.1016/j.neucom.2023.127063 | rotary position embeddings |
| Tung, F. & Mori, G. 2019. Similarity-preserving knowledge distillation. Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 1365-1374. https://doi.org/10.1109/iccv.2019.00145 | similarity-preserving distillation (the relational loss of the student) |
| van den Oord, A., Li, Y. & Vinyals, O. 2018. Representation learning with contrastive predictive coding. arXiv preprint arXiv:1807.03748. | InfoNCE (the contrastive loss of the student) |
| Weininger, D. 1988. SMILES, a chemical language and information system. 1. Introduction to methodology and encoding rules. Journal of Chemical Information and Computer Sciences, 28, 31-36. https://doi.org/10.1021/ci00057a005 | SMILES |
| Zhang, B. & Sennrich, R. 2019. Root mean square layer normalization. Advances in Neural Information Processing Systems (NeurIPS). arXiv preprint arXiv:1910.07467. | RMSNorm |
| Zhang, R., Wu, H., Liu, C., Yang, Q., Xiu, Y., Li, K. et al. 2025. PepLand: a large-scale pre-trained peptide representation model for a comprehensive landscape of both canonical and non-canonical amino acids. Briefings in Bioinformatics, 26, bbaf367. https://doi.org/10.1093/bib/bbaf367 | PepLand: input model of the teacher; the PL_* benchmarks and their published scores |
