import utils
import Bio.SeqIO

# wt sent by bryce
wt = "MQHTYPAQLMRFGTAARAEHMTIAAAIGALGADEADAIVMDIVPDGERDAWWDDEGFSSSPFTKNAHHAGIVATSVTLGQLQGEQGDKLVSKAAEYFGIACRVNDGLRTTRFVRLFSDALDAKPLTIGHDYEVEFLLATRRVYEPFEAPFNFAPHCGDVSYGRDTVNWPLKRSFPRQLGGFLTIQGADNDAGMVMWDNRPESRAALDEMHAEYRETGAIAALERAAKIMLKPQPGQLTLFQSKNLHAIERCTSTRRTMGLFLIHTEDGWRMFD"


def get_prot(parent):
    ret = Bio.SeqIO.read(f"../data/{parent}.fasta", "fasta")
    ret = str(ret.translate().seq)
    return ret

sadA = get_prot("SadA")
print("Difference from SadA.fasta : ", utils.mut_as_string(wt, sadA, offset=1))

sadX = get_prot("SadX")
parent1 = get_prot("1-VH")
print("Difference from 1-VH.fasta : ", utils.mut_as_string(wt, parent1, offset=1))


x = Bio.SeqIO.read(f"../rosetta/SadA_NSLeu_Corrected_3701_0002.pdb", "pdb-atom")
old_pdb = str(x.seq)
print("Difference from old_pdb : ", utils.mut_as_string(wt, old_pdb, offset=1))


print("Difference between old_pdb and sadA : ", utils.mut_as_string(sadA, old_pdb, offset=1))


x = Bio.SeqIO.read(f"/mnt/scratch/sameer/test_rosetta/SadA_NSLeu_Corrected_3701_best_structure_0044.pdb", "pdb-atom")
rosetta_andres = str(x.seq)
print("Difference between rosetta andres and wt : ", utils.mut_as_string(sadA, old_pdb, offset=1))


