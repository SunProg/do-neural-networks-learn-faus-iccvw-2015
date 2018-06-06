for number in {0..9}
do
cd "$number"
for fold in {0..9}
do
echo "$number $fold"
cd "$fold"
ls -al
cd ..
done
cd ..
done
exit 0
