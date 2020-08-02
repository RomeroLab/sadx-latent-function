# load pdbs
#load SadH_activity_mapped.pdb
load SadH_azid_activity_mapped.pdb

# white background
bg_color white
set depth_cue, 0
set ray_trace_fog, 0

# show protein cartoon with side chains
hide everything
show cartoon, all
select bb, name c+o+n 
show lines, !bb
spectrum b, red_white_blue


# show CA of important residues as balls
select balls, b>80 and name ca
show sphere, balls
set sphere_scale, .4


### cut below here and paste into script ###
set_view (\
     0.163004398,    0.937573433,    0.307208896,\
     0.187359065,   -0.335127532,    0.923355758,\
     0.968670249,   -0.092954695,   -0.230289713,\
     0.000000000,    0.000000000, -162.580703735,\
    18.209487915,   -6.148672104,  -16.709634781,\
   128.179870605,  196.981536865,  -20.000000000 )
### cut above here and paste into script ###


ray 2400,2000
png SadH_activity_mapped.png, dpi=300

