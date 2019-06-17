echo start

python3 /data/predict_part1.py --model /data/submit_1_0_200.hdf5 --output /data/submit1_1_0_200.pickle
python3 /data/predict_part2.py --model /data/submit_1_0_200.hdf5 --output /data/submit2_1_0_200.pickle
python3 /data/predict_part3.py --model /data/submit_1_0_200.hdf5 --output /data/submit3_1_0_200.pickle
python3 /data/merge.py --input /data/submit1_1_0_200.pickle,/data/submit2_1_0_200.pickle,/data/submit3_1_0_200.pickle --output /data/submit_1_0_200.pickle
rm /data/submit1_1_0_200.pickle /data/submit2_1_0_200.pickle /data/submit3_1_0_200.pickle

python3 /data/predict_part1.py --model /data/submit_1_20_200.hdf5 --output /data/submit1_1_20_200.pickle
python3 /data/predict_part2.py --model /data/submit_1_20_200.hdf5 --output /data/submit2_1_20_200.pickle
python3 /data/predict_part3.py --model /data/submit_1_20_200.hdf5 --output /data/submit3_1_20_200.pickle
python3 /data/merge.py --input /data/submit1_1_20_200.pickle,/data/submit2_1_20_200.pickle,/data/submit3_1_20_200.pickle --output /data/submit_1_20_200.pickle
rm /data/submit1_1_20_200.pickle /data/submit2_1_20_200.pickle /data/submit3_1_20_200.pickle

python3 /data/predict_part1.py --model /data/submit_1_40_200.hdf5 --output /data/submit1_1_40_200.pickle
python3 /data/predict_part2.py --model /data/submit_1_40_200.hdf5 --output /data/submit2_1_40_200.pickle
python3 /data/predict_part3.py --model /data/submit_1_40_200.hdf5 --output /data/submit3_1_40_200.pickle
python3 /data/merge.py --input /data/submit1_1_40_200.pickle,/data/submit2_1_40_200.pickle,/data/submit3_1_40_200.pickle --output /data/submit_1_40_200.pickle
rm /data/submit1_1_40_200.pickle /data/submit2_1_40_200.pickle /data/submit3_1_40_200.pickle

python3 /data/predict_part1.py --model /data/submit_1_0_60.hdf5 --output /data/submit1_1_0_60.pickle
python3 /data/predict_part2.py --model /data/submit_1_0_60.hdf5 --output /data/submit2_1_0_60.pickle
python3 /data/predict_part3.py --model /data/submit_1_0_60.hdf5 --output /data/submit3_1_0_60.pickle
python3 /data/merge.py --input /data/submit1_1_0_60.pickle,/data/submit2_1_0_60.pickle,/data/submit3_1_0_60.pickle --output /data/submit_1_0_60.pickle
rm /data/submit1_1_0_60.pickle /data/submit2_1_0_60.pickle /data/submit3_1_0_60.pickle

python3 /data/ensemble.py --input /data/submit_1_0_200.pickle,/data/submit_1_20_200.pickle,/data/submit_1_40_200.pickle,/data/submit_1_0_60.pickle --output /data/part1.pickle
rm /data/submit_1_0_200.pickle /data/submit_1_20_200.pickle /data/submit_1_40_200.pickle /data/submit_1_0_60.pickle

echo 1/10

python3 /data/predict_part1.py --model /data/submit_2_0_200.hdf5 --output /data/submit1_2_0_200.pickle
python3 /data/predict_part2.py --model /data/submit_2_0_200.hdf5 --output /data/submit2_2_0_200.pickle
python3 /data/predict_part3.py --model /data/submit_2_0_200.hdf5 --output /data/submit3_2_0_200.pickle
python3 /data/merge.py --input /data/submit1_2_0_200.pickle,/data/submit2_2_0_200.pickle,/data/submit3_2_0_200.pickle --output /data/submit_2_0_200.pickle
rm /data/submit1_2_0_200.pickle /data/submit2_2_0_200.pickle /data/submit3_2_0_200.pickle

python3 /data/predict_part1.py --model /data/submit_2_20_200.hdf5 --output /data/submit1_2_20_200.pickle
python3 /data/predict_part2.py --model /data/submit_2_20_200.hdf5 --output /data/submit2_2_20_200.pickle
python3 /data/predict_part3.py --model /data/submit_2_20_200.hdf5 --output /data/submit3_2_20_200.pickle
python3 /data/merge.py --input /data/submit1_2_20_200.pickle,/data/submit2_2_20_200.pickle,/data/submit3_2_20_200.pickle --output /data/submit_2_20_200.pickle
rm /data/submit1_2_20_200.pickle /data/submit2_2_20_200.pickle /data/submit3_2_20_200.pickle

python3 /data/predict_part1.py --model /data/submit_2_40_200.hdf5 --output /data/submit1_2_40_200.pickle
python3 /data/predict_part2.py --model /data/submit_2_40_200.hdf5 --output /data/submit2_2_40_200.pickle
python3 /data/predict_part3.py --model /data/submit_2_40_200.hdf5 --output /data/submit3_2_40_200.pickle
python3 /data/merge.py --input /data/submit1_2_40_200.pickle,/data/submit2_2_40_200.pickle,/data/submit3_2_40_200.pickle --output /data/submit_2_40_200.pickle
rm /data/submit1_2_40_200.pickle /data/submit2_2_40_200.pickle /data/submit3_2_40_200.pickle

python3 /data/predict_part1.py --model /data/submit_2_0_60.hdf5 --output /data/submit1_2_0_60.pickle
python3 /data/predict_part2.py --model /data/submit_2_0_60.hdf5 --output /data/submit2_2_0_60.pickle
python3 /data/predict_part3.py --model /data/submit_2_0_60.hdf5 --output /data/submit3_2_0_60.pickle
python3 /data/merge.py --input /data/submit1_2_0_60.pickle,/data/submit2_2_0_60.pickle,/data/submit3_2_0_60.pickle --output /data/submit_2_0_60.pickle
rm /data/submit1_2_0_60.pickle /data/submit2_2_0_60.pickle /data/submit3_2_0_60.pickle

python3 /data/ensemble.py --input /data/submit_2_0_200.pickle,/data/submit_2_20_200.pickle,/data/submit_2_40_200.pickle,/data/submit_2_0_60.pickle --output /data/part2.pickle
rm /data/submit_2_0_200.pickle /data/submit_2_20_200.pickle /data/submit_2_40_200.pickle /data/submit_2_0_60.pickle

echo 2/10

python3 /data/predict_part1.py --model /data/submit_3_0_200.hdf5 --output /data/submit1_3_0_200.pickle
python3 /data/predict_part2.py --model /data/submit_3_0_200.hdf5 --output /data/submit2_3_0_200.pickle
python3 /data/predict_part3.py --model /data/submit_3_0_200.hdf5 --output /data/submit3_3_0_200.pickle
python3 /data/merge.py --input /data/submit1_3_0_200.pickle,/data/submit2_3_0_200.pickle,/data/submit3_3_0_200.pickle --output /data/submit_3_0_200.pickle
rm /data/submit1_3_0_200.pickle /data/submit2_3_0_200.pickle /data/submit3_3_0_200.pickle

python3 /data/predict_part1.py --model /data/submit_3_20_200.hdf5 --output /data/submit1_3_20_200.pickle
python3 /data/predict_part2.py --model /data/submit_3_20_200.hdf5 --output /data/submit2_3_20_200.pickle
python3 /data/predict_part3.py --model /data/submit_3_20_200.hdf5 --output /data/submit3_3_20_200.pickle
python3 /data/merge.py --input /data/submit1_3_20_200.pickle,/data/submit2_3_20_200.pickle,/data/submit3_3_20_200.pickle --output /data/submit_3_20_200.pickle
rm /data/submit1_3_20_200.pickle /data/submit2_3_20_200.pickle /data/submit3_3_20_200.pickle

python3 /data/predict_part1.py --model /data/submit_3_40_200.hdf5 --output /data/submit1_3_40_200.pickle
python3 /data/predict_part2.py --model /data/submit_3_40_200.hdf5 --output /data/submit2_3_40_200.pickle
python3 /data/predict_part3.py --model /data/submit_3_40_200.hdf5 --output /data/submit3_3_40_200.pickle
python3 /data/merge.py --input /data/submit1_3_40_200.pickle,/data/submit2_3_40_200.pickle,/data/submit3_3_40_200.pickle --output /data/submit_3_40_200.pickle
rm /data/submit1_3_40_200.pickle /data/submit2_3_40_200.pickle /data/submit3_3_40_200.pickle

python3 /data/predict_part1.py --model /data/submit_3_0_60.hdf5 --output /data/submit1_3_0_60.pickle
python3 /data/predict_part2.py --model /data/submit_3_0_60.hdf5 --output /data/submit2_3_0_60.pickle
python3 /data/predict_part3.py --model /data/submit_3_0_60.hdf5 --output /data/submit3_3_0_60.pickle
python3 /data/merge.py --input /data/submit1_3_0_60.pickle,/data/submit2_3_0_60.pickle,/data/submit3_3_0_60.pickle --output /data/submit_3_0_60.pickle
rm /data/submit1_3_0_60.pickle /data/submit2_3_0_60.pickle /data/submit3_3_0_60.pickle

python3 /data/ensemble.py --input /data/submit_3_0_200.pickle,/data/submit_3_20_200.pickle,/data/submit_3_40_200.pickle,/data/submit_3_0_60.pickle --output /data/part3.pickle
rm /data/submit_3_0_200.pickle /data/submit_3_20_200.pickle /data/submit_3_40_200.pickle /data/submit_3_0_60.pickle

echo 3/10

python3 /data/predict_part1.py --model /data/submit_4_0_200.hdf5 --output /data/submit1_4_0_200.pickle
python3 /data/predict_part2.py --model /data/submit_4_0_200.hdf5 --output /data/submit2_4_0_200.pickle
python3 /data/predict_part3.py --model /data/submit_4_0_200.hdf5 --output /data/submit3_4_0_200.pickle
python3 /data/merge.py --input /data/submit1_4_0_200.pickle,/data/submit2_4_0_200.pickle,/data/submit3_4_0_200.pickle --output /data/submit_4_0_200.pickle
rm /data/submit1_4_0_200.pickle /data/submit2_4_0_200.pickle /data/submit3_4_0_200.pickle

python3 /data/predict_part1.py --model /data/submit_4_20_200.hdf5 --output /data/submit1_4_20_200.pickle
python3 /data/predict_part2.py --model /data/submit_4_20_200.hdf5 --output /data/submit2_4_20_200.pickle
python3 /data/predict_part3.py --model /data/submit_4_20_200.hdf5 --output /data/submit3_4_20_200.pickle
python3 /data/merge.py --input /data/submit1_4_20_200.pickle,/data/submit2_4_20_200.pickle,/data/submit3_4_20_200.pickle --output /data/submit_4_20_200.pickle
rm /data/submit1_4_20_200.pickle /data/submit2_4_20_200.pickle /data/submit3_4_20_200.pickle

python3 /data/predict_part1.py --model /data/submit_4_40_200.hdf5 --output /data/submit1_4_40_200.pickle
python3 /data/predict_part2.py --model /data/submit_4_40_200.hdf5 --output /data/submit2_4_40_200.pickle
python3 /data/predict_part3.py --model /data/submit_4_40_200.hdf5 --output /data/submit3_4_40_200.pickle
python3 /data/merge.py --input /data/submit1_4_40_200.pickle,/data/submit2_4_40_200.pickle,/data/submit3_4_40_200.pickle --output /data/submit_4_40_200.pickle
rm /data/submit1_4_40_200.pickle /data/submit2_4_40_200.pickle /data/submit3_4_40_200.pickle

python3 /data/predict_part1.py --model /data/submit_4_0_60.hdf5 --output /data/submit1_4_0_60.pickle
python3 /data/predict_part2.py --model /data/submit_4_0_60.hdf5 --output /data/submit2_4_0_60.pickle
python3 /data/predict_part3.py --model /data/submit_4_0_60.hdf5 --output /data/submit3_4_0_60.pickle
python3 /data/merge.py --input /data/submit1_4_0_60.pickle,/data/submit2_4_0_60.pickle,/data/submit3_4_0_60.pickle --output /data/submit_4_0_60.pickle
rm /data/submit1_4_0_60.pickle /data/submit2_4_0_60.pickle /data/submit3_4_0_60.pickle

python3 /data/ensemble.py --input /data/submit_4_0_200.pickle,/data/submit_4_20_200.pickle,/data/submit_4_40_200.pickle,/data/submit_4_0_60.pickle --output /data/part4.pickle
rm /data/submit_4_0_200.pickle /data/submit_4_20_200.pickle /data/submit_4_40_200.pickle /data/submit_4_0_60.pickle

echo 4/10

python3 /data/predict_part1.py --model /data/submit_5_0_200.hdf5 --output /data/submit1_5_0_200.pickle
python3 /data/predict_part2.py --model /data/submit_5_0_200.hdf5 --output /data/submit2_5_0_200.pickle
python3 /data/predict_part3.py --model /data/submit_5_0_200.hdf5 --output /data/submit3_5_0_200.pickle
python3 /data/merge.py --input /data/submit1_5_0_200.pickle,/data/submit2_5_0_200.pickle,/data/submit3_5_0_200.pickle --output /data/submit_5_0_200.pickle
rm /data/submit1_5_0_200.pickle /data/submit2_5_0_200.pickle /data/submit3_5_0_200.pickle

python3 /data/predict_part1.py --model /data/submit_5_20_200.hdf5 --output /data/submit1_5_20_200.pickle
python3 /data/predict_part2.py --model /data/submit_5_20_200.hdf5 --output /data/submit2_5_20_200.pickle
python3 /data/predict_part3.py --model /data/submit_5_20_200.hdf5 --output /data/submit3_5_20_200.pickle
python3 /data/merge.py --input /data/submit1_5_20_200.pickle,/data/submit2_5_20_200.pickle,/data/submit3_5_20_200.pickle --output /data/submit_5_20_200.pickle
rm /data/submit1_5_20_200.pickle /data/submit2_5_20_200.pickle /data/submit3_5_20_200.pickle

python3 /data/predict_part1.py --model /data/submit_5_40_200.hdf5 --output /data/submit1_5_40_200.pickle
python3 /data/predict_part2.py --model /data/submit_5_40_200.hdf5 --output /data/submit2_5_40_200.pickle
python3 /data/predict_part3.py --model /data/submit_5_40_200.hdf5 --output /data/submit3_5_40_200.pickle
python3 /data/merge.py --input /data/submit1_5_40_200.pickle,/data/submit2_5_40_200.pickle,/data/submit3_5_40_200.pickle --output /data/submit_5_40_200.pickle
rm /data/submit1_5_40_200.pickle /data/submit2_5_40_200.pickle /data/submit3_5_40_200.pickle

python3 /data/predict_part1.py --model /data/submit_5_0_60.hdf5 --output /data/submit1_5_0_60.pickle
python3 /data/predict_part2.py --model /data/submit_5_0_60.hdf5 --output /data/submit2_5_0_60.pickle
python3 /data/predict_part3.py --model /data/submit_5_0_60.hdf5 --output /data/submit3_5_0_60.pickle
python3 /data/merge.py --input /data/submit1_5_0_60.pickle,/data/submit2_5_0_60.pickle,/data/submit3_5_0_60.pickle --output /data/submit_5_0_60.pickle
rm /data/submit1_5_0_60.pickle /data/submit2_5_0_60.pickle /data/submit3_5_0_60.pickle

python3 /data/ensemble.py --input /data/submit_5_0_200.pickle,/data/submit_5_20_200.pickle,/data/submit_5_40_200.pickle,/data/submit_5_0_60.pickle --output /data/part5.pickle
rm /data/submit_5_0_200.pickle /data/submit_5_20_200.pickle /data/submit_5_40_200.pickle /data/submit_5_0_60.pickle

echo 5/10

python3 /data/predict_part1.py --model /data/submit_6_0_200.hdf5 --output /data/submit1_6_0_200.pickle
python3 /data/predict_part2.py --model /data/submit_6_0_200.hdf5 --output /data/submit2_6_0_200.pickle
python3 /data/predict_part3.py --model /data/submit_6_0_200.hdf5 --output /data/submit3_6_0_200.pickle
python3 /data/merge.py --input /data/submit1_6_0_200.pickle,/data/submit2_6_0_200.pickle,/data/submit3_6_0_200.pickle --output /data/submit_6_0_200.pickle
rm /data/submit1_6_0_200.pickle /data/submit2_6_0_200.pickle /data/submit3_6_0_200.pickle

python3 /data/predict_part1.py --model /data/submit_6_20_200.hdf5 --output /data/submit1_6_20_200.pickle
python3 /data/predict_part2.py --model /data/submit_6_20_200.hdf5 --output /data/submit2_6_20_200.pickle
python3 /data/predict_part3.py --model /data/submit_6_20_200.hdf5 --output /data/submit3_6_20_200.pickle
python3 /data/merge.py --input /data/submit1_6_20_200.pickle,/data/submit2_6_20_200.pickle,/data/submit3_6_20_200.pickle --output /data/submit_6_20_200.pickle
rm /data/submit1_6_20_200.pickle /data/submit2_6_20_200.pickle /data/submit3_6_20_200.pickle

python3 /data/predict_part1.py --model /data/submit_6_40_200.hdf5 --output /data/submit1_6_40_200.pickle
python3 /data/predict_part2.py --model /data/submit_6_40_200.hdf5 --output /data/submit2_6_40_200.pickle
python3 /data/predict_part3.py --model /data/submit_6_40_200.hdf5 --output /data/submit3_6_40_200.pickle
python3 /data/merge.py --input /data/submit1_6_40_200.pickle,/data/submit2_6_40_200.pickle,/data/submit3_6_40_200.pickle --output /data/submit_6_40_200.pickle
rm /data/submit1_6_40_200.pickle /data/submit2_6_40_200.pickle /data/submit3_6_40_200.pickle

python3 /data/predict_part1.py --model /data/submit_6_0_60.hdf5 --output /data/submit1_6_0_60.pickle
python3 /data/predict_part2.py --model /data/submit_6_0_60.hdf5 --output /data/submit2_6_0_60.pickle
python3 /data/predict_part3.py --model /data/submit_6_0_60.hdf5 --output /data/submit3_6_0_60.pickle
python3 /data/merge.py --input /data/submit1_6_0_60.pickle,/data/submit2_6_0_60.pickle,/data/submit3_6_0_60.pickle --output /data/submit_6_0_60.pickle
rm /data/submit1_6_0_60.pickle /data/submit2_6_0_60.pickle /data/submit3_6_0_60.pickle

python3 /data/ensemble.py --input /data/submit_6_0_200.pickle,/data/submit_6_20_200.pickle,/data/submit_6_40_200.pickle,/data/submit_6_0_60.pickle --output /data/part6.pickle
rm /data/submit_6_0_200.pickle /data/submit_6_20_200.pickle /data/submit_6_40_200.pickle /data/submit_6_0_60.pickle

echo 6/10

python3 /data/predict_part1.py --model /data/submit_7_0_200.hdf5 --output /data/submit1_7_0_200.pickle
python3 /data/predict_part2.py --model /data/submit_7_0_200.hdf5 --output /data/submit2_7_0_200.pickle
python3 /data/predict_part3.py --model /data/submit_7_0_200.hdf5 --output /data/submit3_7_0_200.pickle
python3 /data/merge.py --input /data/submit1_7_0_200.pickle,/data/submit2_7_0_200.pickle,/data/submit3_7_0_200.pickle --output /data/submit_7_0_200.pickle
rm /data/submit1_7_0_200.pickle /data/submit2_7_0_200.pickle /data/submit3_7_0_200.pickle

python3 /data/predict_part1.py --model /data/submit_7_20_200.hdf5 --output /data/submit1_7_20_200.pickle
python3 /data/predict_part2.py --model /data/submit_7_20_200.hdf5 --output /data/submit2_7_20_200.pickle
python3 /data/predict_part3.py --model /data/submit_7_20_200.hdf5 --output /data/submit3_7_20_200.pickle
python3 /data/merge.py --input /data/submit1_7_20_200.pickle,/data/submit2_7_20_200.pickle,/data/submit3_7_20_200.pickle --output /data/submit_7_20_200.pickle
rm /data/submit1_7_20_200.pickle /data/submit2_7_20_200.pickle /data/submit3_7_20_200.pickle

python3 /data/predict_part1.py --model /data/submit_7_40_200.hdf5 --output /data/submit1_7_40_200.pickle
python3 /data/predict_part2.py --model /data/submit_7_40_200.hdf5 --output /data/submit2_7_40_200.pickle
python3 /data/predict_part3.py --model /data/submit_7_40_200.hdf5 --output /data/submit3_7_40_200.pickle
python3 /data/merge.py --input /data/submit1_7_40_200.pickle,/data/submit2_7_40_200.pickle,/data/submit3_7_40_200.pickle --output /data/submit_7_40_200.pickle
rm /data/submit1_7_40_200.pickle /data/submit2_7_40_200.pickle /data/submit3_7_40_200.pickle

python3 /data/predict_part1.py --model /data/submit_7_0_60.hdf5 --output /data/submit1_7_0_60.pickle
python3 /data/predict_part2.py --model /data/submit_7_0_60.hdf5 --output /data/submit2_7_0_60.pickle
python3 /data/predict_part3.py --model /data/submit_7_0_60.hdf5 --output /data/submit3_7_0_60.pickle
python3 /data/merge.py --input /data/submit1_7_0_60.pickle,/data/submit2_7_0_60.pickle,/data/submit3_7_0_60.pickle --output /data/submit_7_0_60.pickle
rm /data/submit1_7_0_60.pickle /data/submit2_7_0_60.pickle /data/submit3_7_0_60.pickle

python3 /data/ensemble.py --input /data/submit_7_0_200.pickle,/data/submit_7_20_200.pickle,/data/submit_7_40_200.pickle,/data/submit_7_0_60.pickle --output /data/part7.pickle
rm /data/submit_7_0_200.pickle /data/submit_7_20_200.pickle /data/submit_7_40_200.pickle /data/submit_7_0_60.pickle

echo 7/10

python3 /data/predict_part1.py --model /data/submit_8_0_200.hdf5 --output /data/submit1_8_0_200.pickle
python3 /data/predict_part2.py --model /data/submit_8_0_200.hdf5 --output /data/submit2_8_0_200.pickle
python3 /data/predict_part3.py --model /data/submit_8_0_200.hdf5 --output /data/submit3_8_0_200.pickle
python3 /data/merge.py --input /data/submit1_8_0_200.pickle,/data/submit2_8_0_200.pickle,/data/submit3_8_0_200.pickle --output /data/submit_8_0_200.pickle
rm /data/submit1_8_0_200.pickle /data/submit2_8_0_200.pickle /data/submit3_8_0_200.pickle

python3 /data/predict_part1.py --model /data/submit_8_20_200.hdf5 --output /data/submit1_8_20_200.pickle
python3 /data/predict_part2.py --model /data/submit_8_20_200.hdf5 --output /data/submit2_8_20_200.pickle
python3 /data/predict_part3.py --model /data/submit_8_20_200.hdf5 --output /data/submit3_8_20_200.pickle
python3 /data/merge.py --input /data/submit1_8_20_200.pickle,/data/submit2_8_20_200.pickle,/data/submit3_8_20_200.pickle --output /data/submit_8_20_200.pickle
rm /data/submit1_8_20_200.pickle /data/submit2_8_20_200.pickle /data/submit3_8_20_200.pickle

python3 /data/predict_part1.py --model /data/submit_8_40_200.hdf5 --output /data/submit1_8_40_200.pickle
python3 /data/predict_part2.py --model /data/submit_8_40_200.hdf5 --output /data/submit2_8_40_200.pickle
python3 /data/predict_part3.py --model /data/submit_8_40_200.hdf5 --output /data/submit3_8_40_200.pickle
python3 /data/merge.py --input /data/submit1_8_40_200.pickle,/data/submit2_8_40_200.pickle,/data/submit3_8_40_200.pickle --output /data/submit_8_40_200.pickle
rm /data/submit1_8_40_200.pickle /data/submit2_8_40_200.pickle /data/submit3_8_40_200.pickle

python3 /data/predict_part1.py --model /data/submit_8_0_60.hdf5 --output /data/submit1_8_0_60.pickle
python3 /data/predict_part2.py --model /data/submit_8_0_60.hdf5 --output /data/submit2_8_0_60.pickle
python3 /data/predict_part3.py --model /data/submit_8_0_60.hdf5 --output /data/submit3_8_0_60.pickle
python3 /data/merge.py --input /data/submit1_8_0_60.pickle,/data/submit2_8_0_60.pickle,/data/submit3_8_0_60.pickle --output /data/submit_8_0_60.pickle
rm /data/submit1_8_0_60.pickle /data/submit2_8_0_60.pickle /data/submit3_8_0_60.pickle

python3 /data/ensemble.py --input /data/submit_8_0_200.pickle,/data/submit_8_20_200.pickle,/data/submit_8_40_200.pickle,/data/submit_8_0_60.pickle --output /data/part8.pickle
rm /data/submit_8_0_200.pickle /data/submit_8_20_200.pickle /data/submit_8_40_200.pickle /data/submit_8_0_60.pickle

echo 8/10

python3 /data/predict_part1.py --model /data/submit_9_0_200.hdf5 --output /data/submit1_9_0_200.pickle
python3 /data/predict_part2.py --model /data/submit_9_0_200.hdf5 --output /data/submit2_9_0_200.pickle
python3 /data/predict_part3.py --model /data/submit_9_0_200.hdf5 --output /data/submit3_9_0_200.pickle
python3 /data/merge.py --input /data/submit1_9_0_200.pickle,/data/submit2_9_0_200.pickle,/data/submit3_9_0_200.pickle --output /data/submit_9_0_200.pickle
rm /data/submit1_9_0_200.pickle /data/submit2_9_0_200.pickle /data/submit3_9_0_200.pickle

python3 /data/predict_part1.py --model /data/submit_9_20_200.hdf5 --output /data/submit1_9_20_200.pickle
python3 /data/predict_part2.py --model /data/submit_9_20_200.hdf5 --output /data/submit2_9_20_200.pickle
python3 /data/predict_part3.py --model /data/submit_9_20_200.hdf5 --output /data/submit3_9_20_200.pickle
python3 /data/merge.py --input /data/submit1_9_20_200.pickle,/data/submit2_9_20_200.pickle,/data/submit3_9_20_200.pickle --output /data/submit_9_20_200.pickle
rm /data/submit1_9_20_200.pickle /data/submit2_9_20_200.pickle /data/submit3_9_20_200.pickle

python3 /data/predict_part1.py --model /data/submit_9_40_200.hdf5 --output /data/submit1_9_40_200.pickle
python3 /data/predict_part2.py --model /data/submit_9_40_200.hdf5 --output /data/submit2_9_40_200.pickle
python3 /data/predict_part3.py --model /data/submit_9_40_200.hdf5 --output /data/submit3_9_40_200.pickle
python3 /data/merge.py --input /data/submit1_9_40_200.pickle,/data/submit2_9_40_200.pickle,/data/submit3_9_40_200.pickle --output /data/submit_9_40_200.pickle
rm /data/submit1_9_40_200.pickle /data/submit2_9_40_200.pickle /data/submit3_9_40_200.pickle

python3 /data/predict_part1.py --model /data/submit_9_0_60.hdf5 --output /data/submit1_9_0_60.pickle
python3 /data/predict_part2.py --model /data/submit_9_0_60.hdf5 --output /data/submit2_9_0_60.pickle
python3 /data/predict_part3.py --model /data/submit_9_0_60.hdf5 --output /data/submit3_9_0_60.pickle
python3 /data/merge.py --input /data/submit1_9_0_60.pickle,/data/submit2_9_0_60.pickle,/data/submit3_9_0_60.pickle --output /data/submit_9_0_60.pickle
rm /data/submit1_9_0_60.pickle /data/submit2_9_0_60.pickle /data/submit3_9_0_60.pickle

python3 /data/ensemble.py --input /data/submit_9_0_200.pickle,/data/submit_9_20_200.pickle,/data/submit_9_40_200.pickle,/data/submit_9_0_60.pickle --output /data/part9.pickle
rm /data/submit_9_0_200.pickle /data/submit_9_20_200.pickle /data/submit_9_40_200.pickle /data/submit_9_0_60.pickle

echo 9/10

python3 /data/predict_part1.py --model /data/submit_10_0_200.hdf5 --output /data/submit1_10_0_200.pickle
python3 /data/predict_part2.py --model /data/submit_10_0_200.hdf5 --output /data/submit2_10_0_200.pickle
python3 /data/predict_part3.py --model /data/submit_10_0_200.hdf5 --output /data/submit3_10_0_200.pickle
python3 /data/merge.py --input /data/submit1_10_0_200.pickle,/data/submit2_10_0_200.pickle,/data/submit3_10_0_200.pickle --output /data/submit_10_0_200.pickle
rm /data/submit1_10_0_200.pickle /data/submit2_10_0_200.pickle /data/submit3_10_0_200.pickle

python3 /data/predict_part1.py --model /data/submit_10_20_200.hdf5 --output /data/submit1_10_20_200.pickle
python3 /data/predict_part2.py --model /data/submit_10_20_200.hdf5 --output /data/submit2_10_20_200.pickle
python3 /data/predict_part3.py --model /data/submit_10_20_200.hdf5 --output /data/submit3_10_20_200.pickle
python3 /data/merge.py --input /data/submit1_10_20_200.pickle,/data/submit2_10_20_200.pickle,/data/submit3_10_20_200.pickle --output /data/submit_10_20_200.pickle
rm /data/submit1_10_20_200.pickle /data/submit2_10_20_200.pickle /data/submit3_10_20_200.pickle

python3 /data/predict_part1.py --model /data/submit_10_40_200.hdf5 --output /data/submit1_10_40_200.pickle
python3 /data/predict_part2.py --model /data/submit_10_40_200.hdf5 --output /data/submit2_10_40_200.pickle
python3 /data/predict_part3.py --model /data/submit_10_40_200.hdf5 --output /data/submit3_10_40_200.pickle
python3 /data/merge.py --input /data/submit1_10_40_200.pickle,/data/submit2_10_40_200.pickle,/data/submit3_10_40_200.pickle --output /data/submit_10_40_200.pickle
rm /data/submit1_10_40_200.pickle /data/submit2_10_40_200.pickle /data/submit3_10_40_200.pickle

python3 /data/predict_part1.py --model /data/submit_10_0_60.hdf5 --output /data/submit1_10_0_60.pickle
python3 /data/predict_part2.py --model /data/submit_10_0_60.hdf5 --output /data/submit2_10_0_60.pickle
python3 /data/predict_part3.py --model /data/submit_10_0_60.hdf5 --output /data/submit3_10_0_60.pickle
python3 /data/merge.py --input /data/submit1_10_0_60.pickle,/data/submit2_10_0_60.pickle,/data/submit3_10_0_60.pickle --output /data/submit_10_0_60.pickle
rm /data/submit1_10_0_60.pickle /data/submit2_10_0_60.pickle /data/submit3_10_0_60.pickle

python3 /data/ensemble.py --input /data/submit_10_0_200.pickle,/data/submit_10_20_200.pickle,/data/submit_10_40_200.pickle,/data/submit_10_0_60.pickle --output /data/part10.pickle
rm /data/submit_10_0_200.pickle /data/submit_10_20_200.pickle /data/submit_10_40_200.pickle /data/submit_10_0_60.pickle

echo 10/10

python3 /data/ensemble.py --input /data/part1.pickle,/data/part2.pickle,/data/part3.pickle,/data/part4.pickle,/data/part5.pickle --output /data/submit1.pickle
rm /data/part1.pickle /data/part2.pickle /data/part3.pickle /data/part4.pickle /data/part5.pickle

python3 /data/ensemble.py --input /data/part6.pickle,/data/part7.pickle,/data/part8.pickle,/data/part9.pickle,/data/part10.pickle --output /data/submit2.pickle
rm /data/part6.pickle /data/part7.pickle /data/part8.pickle /data/part9.pickle /data/part10.pickle

python3 /data/ensemble.py --input /data/submit1.pickle,/data/submit2.pickle --output /data/result/result.txt
rm /data/submit1.pickle /data/submit2.pickle

echo end