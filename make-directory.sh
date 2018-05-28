mkdir result
cd result
for number in {0..9}
do
mkdir "$number"
cd "$number"
for fold in {0..9}
do
mkdir "$fold"
done
cd ..
done
exit 0