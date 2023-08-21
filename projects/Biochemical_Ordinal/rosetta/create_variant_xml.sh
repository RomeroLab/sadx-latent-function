#!/bin/bash

# Read in a mutant string like this
# and print out the xml for rosetta to mutate

if [[ $# -le 1 ]] ; then
    echo 'Usage: ./$0 VARIANT CHAIN'
    echo 'Example: ./$0 Z28G.X31G.B157G A'
    exit 1
fi

VARIANT=$1
CHAIN=$2

# Save the IFS variable so that the read commands below
# can have spaces in them
OLD_IFS=$IFS
IFS=''

read -r -d '' OUTPUT_START << 'EOF' 
<ROSETTASCRIPTS>

		<SCOREFXNS>
		</SCOREFXNS>
		<RESIDUE_SELECTORS>
EOF

read -r -d '' OUTPUT_MIDDLE_TOP << 'EOF'
			<Not name="rest" selector="surrounding"/>
		</RESIDUE_SELECTORS>
		<TASKOPERATIONS>
			<OperateOnResidueSubset name="repack_res" selector="surrounding" >
				<RestrictToRepackingRLT/>
			</OperateOnResidueSubset>
			<OperateOnResidueSubset name="no_repack" selector="rest" >
				<PreventRepackingRLT/>
			</OperateOnResidueSubset>
		</TASKOPERATIONS>
		<MOVERS>
EOF

read -r -d '' OUTPUT_MIDDLE_BOTTOM << 'EOF'
			<FastRelax name="relax" scorefxn="REF2015" task_operations="repack_res,no_repack" min_type="lbfgs_armijo_nonmonotone">
			</FastRelax>
		</MOVERS>
		<PROTOCOLS>
EOF

read -r -d '' OUTPUT_END << 'EOF'
			<Add mover_name="relax"/>
		</PROTOCOLS>


</ROSETTASCRIPTS>
EOF

IFS=$OLD_IFS
# Restore IFS 

# Now print out the xml file
awk -v variant="$VARIANT" \
    -v output_start="$OUTPUT_START" \
    -v output_middle_top="$OUTPUT_MIDDLE_TOP" \
    -v output_middle_bottom="$OUTPUT_MIDDLE_BOTTOM" \
    -v output_end="$OUTPUT_END" \
    -v chain="$CHAIN" \
    'BEGIN{
       aa_map["A"] =  "ALA"
       aa_map["C"] =  "CYS"
       aa_map["D"] =  "ASP"
       aa_map["E"] =  "GLU"
       aa_map["F"] =  "PHE"
       aa_map["G"] =  "GLY"
       aa_map["H"] =  "HIS"
       aa_map["I"] =  "ILE"
       aa_map["K"] =  "LYS"
       aa_map["L"] =  "LEU"
       aa_map["M"] =  "MET"
       aa_map["N"] =  "ASN"
       aa_map["P"] =  "PRO"
       aa_map["Q"] =  "GLN"
       aa_map["R"] =  "ARG"
       aa_map["S"] =  "SER"
       aa_map["T"] =  "THR"
       aa_map["V"] =  "VAL"
       aa_map["W"] =  "TRP"
       aa_map["Y"] =  "TYR"
 
       nvar = split(variant, variant_arr, ".")
       for (i=1; i <= nvar; i++) { 
         v = variant_arr[i]
         if (length(v) < 3){
           print "ERROR: length(variant) < 3 : " v > "/dev/stderr"
           exit 1
         }
         vparentaa = substr(v, 1, 1)
         if (vparentaa !~ "[A-Z]") {
           print "ERROR: Parent AA is not a capital letter : " v > "/dev/stderr"
           exit 1
         }
         vidx = substr(v, 2, length(v)-2)
         if (vidx !~ "^[[:digit:]]+$") {
           print "ERROR: Residue index is not a number : "  v > "/dev/stderr"
           exit 1
         }
         vidx_arr[i] = vidx
         vnewaa = substr(v, length(v), 1)
         if (vnewaa !~ "[A-Z]") {
           print "ERROR: New AA is not a capital letter : "  v > "/dev/stderr"
           exit 1
         }
         vnewaa_arr[i] = vnewaa
         if (!(vnewaa in aa_map)) {
           print "ERROR: New Amino Acid is not in amino acid map :" v > "/dev/stderr"
           exit 1
         }
         vnewaa_full_arr[i] = aa_map[vnewaa]
       }
       joined_idxs = sep = ""
       for (i=1; i <= nvar; i++) {
         joined_idxs = joined_idxs sep vidx_arr[i]
         sep = ","
       }
       printf output_start
       print "\t\t\t" "<Neighborhood name=\"surrounding\" resnums=\"" joined_idxs "\" distance=\"100.0\"/>"
       printf output_middle_top
       for (i=1; i <= nvar; i++) {
         print "\t\t\t" "<MutateResidue name=\"mutant" i "\" target=\"" vidx_arr[i] chain "\" new_res=\"" vnewaa_full_arr[i] "\"/>"
       }
       printf output_middle_bottom
       for (i=1; i <= nvar; i++)  {
         print "\t\t\t" "<Add mover_name=\"mutant" i "\"/>"
       }
       printf output_end
     }'

