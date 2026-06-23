cd ~/code

wget http://eddylab.org/software/hmmer/hmmer-3.4.tar.gz

tar zxf hmmer-3.4.tar.gz

cd hmmer-3.4/
./configure
make
sudo make install
cd easel
sudo make install
