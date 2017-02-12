mkdir 256x256_random_iterations0000to0031
mv -i {source,destination}_256x256_random_iteration000[0123456789].csv 256x256_random_iterations0000to0031
for ix in {10..31}
do
    mv -i {source,destination}_256x256_random_iteration00$ix.csv 256x256_random_iterations0000to0031
done

mkdir 256x256_random_iterations0032to0063
for ix in {32..63}
do
    mv -i {source,destination}_256x256_random_iteration00$ix.csv 256x256_random_iterations0032to0063
done

mkdir 256x256_random_iterations0064to0095
for ix in {64..95}
do
    mv -i {source,destination}_256x256_random_iteration00$ix.csv 256x256_random_iterations0064to0095
done

mkdir 256x256_random_iterations0096to0127
mv -i {source,destination}_256x256_random_iteration009[6789].csv 256x256_random_iterations0096to0127
for ix in {100..127}
do
    mv -i {source,destination}_256x256_random_iteration0$ix.csv 256x256_random_iterations0096to0127
done


mkdir 256x256_random_iterations0128to0159
for ix in {128..159}
do
    mv -i {source,destination}_256x256_random_iteration0$ix.csv 256x256_random_iterations0128to0159
done

mkdir 256x256_random_iterations0160to0191
for ix in {160..191}
do
    mv -i {source,destination}_256x256_random_iteration0$ix.csv 256x256_random_iterations0160to0191
done

mkdir 256x256_random_iterations0192to0223
for ix in {192..223}
do
    mv -i {source,destination}_256x256_random_iteration0$ix.csv 256x256_random_iterations0192to0223
done

mkdir 256x256_random_iterations0224to0255
for ix in {224..255}
do
    mv -i {source,destination}_256x256_random_iteration0$ix.csv 256x256_random_iterations0224to0255
done

parallel --xapply -v zip ::: -r ::: -9 ::: -q ::: 256x256_random_iterations0000to0031.zip 256x256_random_iterations0032to0063.zip 256x256_random_iterations0064to0095.zip 256x256_random_iterations0096to0127.zip 256x256_random_iterations0128to0159.zip 256x256_random_iterations0160to0191.zip 256x256_random_iterations0192to0223.zip 256x256_random_iterations0224to0255.zip ::: 256x256_random_iterations0000to0031 256x256_random_iterations0032to0063 256x256_random_iterations0064to0095 256x256_random_iterations0096to0127 256x256_random_iterations0128to0159 256x256_random_iterations0160to0191 256x256_random_iterations0192to0223 256x256_random_iterations0224to0255

rm -r 256x256_random_iterations0000to0031
rm -r 256x256_random_iterations0032to0063
rm -r 256x256_random_iterations0064to0095
rm -r 256x256_random_iterations0096to0127
rm -r 256x256_random_iterations0128to0159
rm -r 256x256_random_iterations0160to0191
rm -r 256x256_random_iterations0192to0223
rm -r 256x256_random_iterations0224to0255

ls -sh

