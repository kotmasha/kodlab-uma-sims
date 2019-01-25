# interval experiments

#python launcher.py interval_QAT_dull.yml
#echo "1 of 12 done: interval_QAT_dull"
#python launcher.py interval_QAT_sharp.yml
#echo "2 of 12 done: interval_QAT_sharp"
#python launcher.py interval_EAT_dull.yml
#echo "3 of 12 done: interval_EAT_dull"

python launcher.py interval_EAT_sharp.yml
echo "4 of 12 done: interval_EAT_sharp"
python launcher.py interval_DAT_dull.yml
echo "5 of 12 done: interval_DAT_dull"
python launcher.py interval_DAT_sharp.yml
echo "6 of 12 done: interval_DAT_sharp"

# circle experiments

#python launcher.py circle_QAT_dull.yml
#echo "7 of 12 done: circle_QAT_dull"
#python launcher.py circle_QAT_sharp.yml
#echo "8 of 12 done: circle_QAT_sharp"
#python launcher.py circle_EAT_dull.yml
#echo "9 of 12 done: circle_EAT_dull"

python launcher.py circle_EAT_sharp.yml
echo "10 of 12 done: circle_EAT_sharp"
python launcher.py circle_DAT_dull.yml
echo "11 of 12 done: circle_DAT_dull"
python launcher.py circle_DAT_sharp.yml
echo "12 of 12 done: circle_DAT_sharp"

