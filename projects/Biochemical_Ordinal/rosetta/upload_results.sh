upload_tag="andres_temp_5_onerelax200_rep_1"
smbclient -k //research.drive.wisc.edu/promero2 <<SMBCLIENTCOMMANDS
prompt
cd General/Sameer/SadA_azidation/rosetta/
mkdir $upload_tag
cd $upload_tag
lcd results
mput *.tar.gz
exit
SMBCLIENTCOMMANDS
