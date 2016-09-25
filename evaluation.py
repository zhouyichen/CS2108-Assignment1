from params import *
import csv

CH = 'Color Histogram'
DL = 'Deep Learning'
VW = 'Visual Keyword'
VC = 'Visual Concept'

methods = [CH, DL, VW, VC]

def read_single_results_file(filename, sorted_list=True):
    results = []

    # open the sift feature file for reading
    with open(filename) as f:
        # initialize the CSV reader
        reader = csv.reader(f)

        # loop over the rows in the index
        for row in reader:
			results_raw = row[1:]
			if sorted_list:
				result_list = [[float(results_raw[i]), results_raw[i+1]] for i in range(0, len(results_raw), 2)]
			else:
				result_list = {}
				for i in range(0, len(results_raw), 2):
					result_list[results_raw[i+1]] = float(results_raw[i])
			results.append((row[0], result_list))

        # close the reader
        f.close()
    return results


def get_category_and_id(image_id):
    s = image_id.find("/")
    return (image_id[:s], image_id[s+1:])

def calculate_precision(input_image_id, results):
    category = get_category_and_id(input_image_id)[0]
    correct_count = 0.0
    for res in results:
        if category in res[0]:
            correct_count += 1.0
    return correct_count / TOP_RESULTS_NUMBER

def get_top_resutls(results, number):
    '''
    get the top N results, and combine duplicate images with different category names
    '''
    top_results = []
    index = 0
    top = 0
    while top < number:
        score, image = results[index]
        category, image_id = get_category_and_id(image)
        cat = [category]
        index += 1
        for j in range(index, len(results)):
            next_image = results[j][1]
            next_category, next_id = get_category_and_id(next_image)
            if image_id == next_id:
                cat.append(next_category)
                index += 1
            else:
                top_results.append([cat, score])
                break
        top += 1
    return top_results

def evaluate_MAP_for_single_feature(result_file):
	all_results = read_single_results_file(result_file)
	evaluation_results = []
	sum_MAP = 0
	cat_map = 0
	cat_count = 0
	for image_id, res in all_results:

		top_results = get_top_resutls(res, TOP_RESULTS_NUMBER)
		precision = calculate_precision(image_id, top_results)
		sum_MAP += precision
		cat_map += precision
		cat_count += 1
		if cat_count == NUMBER_OF_TEST_IN_EACH_CAT:
			evaluation_results.append([get_category_and_id(image_id)[0], cat_map / NUMBER_OF_TEST_IN_EACH_CAT])
			cat_count = 0
			cat_map = 0

	evaluation_results.append(['All', sum_MAP/TOTAL_NUMBER_OF_TESTS])

	return evaluation_results

def run_evalutaion_for_all_single_method():
	output = open("evaluation_single.csv", "w")
	ch_map = evaluate_MAP_for_single_feature(ch_results)
	dl_map = evaluate_MAP_for_single_feature(dl_results)
	vw_map = evaluate_MAP_for_single_feature(vw_results)
	vc_map = evaluate_MAP_for_single_feature(vc_results)

	output.write("Categories/Methods,")
	output.write("%s\n" % (",".join(methods)))

	for i in range(NUMBER_OF_CATEGORIES):
		output.write(category_names[i] + ",")
		output.write("%s," % (ch_map[i][1]))
		output.write("%s," % (dl_map[i][1]))
		output.write("%s," % (vw_map[i][1]))
		output.write("%s\n" % (vc_map[i][1]))

	i = 30
	output.write('All' + ",")
	output.write("%s," % (ch_map[i][1]))
	output.write("%s," % (dl_map[i][1]))
	output.write("%s," % (vw_map[i][1]))
	output.write("%s\n" % (vc_map[i][1]))

def combine_results(image_ids, results):
    if len(results) > 1:
		final_result = {}
		for image_id in image_ids:
			score = sum(r[0][image_id] * r[1] for r in results)
	    	final_result[image_id] = score
		return final_result
    else:
        return results[0][0]

def find_weights_for_combination():
	ch_dict = read_single_results_file(ch_results, sorted_list=False)
	dl_dict = read_single_results_file(dl_results, sorted_list=False)
	vw_dict = read_single_results_file(vw_results, sorted_list=False)
	vc_dict = read_single_results_file(vc_results, sorted_list=False)

	small_weights = [1, 2, 4]
	large_weights = [10, 20, 40, 80]

	comb_resulst = []
	image_ids = ch_dict[0][1].keys()

	for ch_weight in small_weights:
		for vw_weight in small_weights:
			for dl_weight in large_weights:
				for vc_weight in large_weights:
					ch_weight = ch_weight / 4
					sum_MAP = 0
					for i in range(TOTAL_NUMBER_OF_TESTS):
						image_id = ch_dict[i][0]
						result_list = []
						result_list.append([ch_dict[i][1], ch_weight])
						result_list.append([dl_dict[i][1], dl_weight])
						result_list.append([vw_dict[i][1], vw_weight])
						result_list.append([vc_dict[i][1], vc_weight])
						combined = combine_results(image_ids, result_list)
						print combined
						combined = sorted([(v, k) for (k, v) in combined.items()])
						top_results = get_top_resutls(combined, TOP_RESULTS_NUMBER)
						precision = calculate_precision(image_id, top_results)
						sum_MAP += precision
					comb_resulst = [[ch_weight, vw_weight, dl_weight, vc_weight], sum_MAP/TOTAL_NUMBER_OF_TESTS]
					print(comb_resulst)
	return comb_resulst

if __name__ == "__main__":
	result = find_weights_for_combination()
	# for i in result:
	# 	print i
