import argparse
import pickle
import heapq


def load_pickle(path):
    with open(path, 'rb') as fin:
        result = pickle.load(fin, encoding='bytes')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Merge the results of parts')
    parser.add_argument('--input', default='/data/part1.pickle,/data/part2.pickle', help='pickles to merge (input, separated by commas)')
    parser.add_argument('--output', default='/data/result.pickle', help='merged result (output, pickle/txt)')
    args = parser.parse_args()
    
    # Merge the results of parts
    pickle_paths = args.input.split(',')
    pickles = [load_pickle(path) for path in pickle_paths]
    output = []
    for index, pic in enumerate(pickles):
        print('part {} len: {}'.format(index, len(pic)))
        pic.sort(key = lambda x: x[0])
        for data in pic:
            video_name = data[0]
            result = data[1]
            output.append((video_name, result))
    print('result len:', len(output))
    print('Merged...')
    
    # Save the merged result
    output_path = args.output
    if output_path.split('.')[-1] == 'txt':
        print('Start sorting...')
        classify_result = [[] for i in range(10034)]
        for i in range(10034):
            top_100 = heapq.nlargest(100, output, key = lambda x: x[1][i])
            classify_result[i] = top_100
            
        with open(output_path, 'w') as fout:
            for i in range(10034):
                output_str = str(i + 1)
                top_100 = classify_result[i]
                for data in top_100:
                    output_str += (' ' + data[0].decode() + '.mp4')
                output_str += '\n'
                fout.write(output_str)
    elif output_path.split('.')[-1] == 'pickle':
        with open(output_path, 'wb') as fout:
            pickle.dump(output, fout)
    else:
        print('Unknown output type')
    print('Saved...')
    