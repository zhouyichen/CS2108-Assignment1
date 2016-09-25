# import the necessary packages
from params import *
from pyimagesearch.color_hist_searcher import *
from pyimagesearch.bow_searcher import *
from pyimagesearch.deep_learning_searcher import *
from Tkinter import *
import tkFileDialog
import tkMessageBox
import os.path
from PIL import Image, ImageTk
from evaluation import get_category_and_id

def combine_results(image_ids, results):
    if len(results) > 1:
        final_result = {}
        for image_id in image_ids:
            score = sum(r[0][image_id] * r[1] for r in results)
            final_result[image_id] = score
        return final_result
    else:
        return results[0][0]

def get_top_resutls(results, number, categories):
    '''
    get the top N results, and combine duplicate images with different category names
    '''
    top_results = []
    index = 0
    top = 0
    while top < number:
        score, image = results[index]
        category, image_id = get_category_and_id(image)
        index += 1
        if category in categories:
            for j in range(index, len(results)):
                next_image = results[j][1]
                next_category, next_id = get_category_and_id(next_image)
                if image_id == next_id:
                    index += 1
                else:
                    top_results.append([score, image])
                    break
            top += 1
    return top_results


class UI_class:
    def __init__(self, master, search_path):
        self.search_path = search_path
        self.master = master
        topframe = Frame(self.master)
        topframe.pack()

        self.color_hist_searcher = ColorHistSearcher(color_hist_train_data)
        self.bow_searcher = BOW_Searcher(bow_train_data)
        self.deep_learning_searcher = DeepLearningSearcher(deep_learning_train_data, visual_concept_train_data)
        self.image_ids = self.bow_searcher.image_ids

        #Buttons
        topspace = Label(topframe).grid(row=0, columnspan=2)
        self.bbutton= Button(topframe, text=" Choose an image ", command=self.browse_query_img)
        self.bbutton.grid(row=1, column=1)
        feature_options = [
            "Color Histogram", 
            "Visual Keywords",
            "Visual Concept",
            "Deep Learning",
        ]
        self.listbox = Listbox(topframe, selectmode=MULTIPLE)
        for feature in feature_options:
            self.listbox.insert(END, feature)
        self.listbox.grid(row=1, column=2)
        self.cbutton = Button(topframe, text=" Search ", command=self.show_results_imgs)
        self.cbutton.grid(row=1, column=3)
        downspace = Label(topframe).grid(row=3, columnspan=4)

        self.master.mainloop()

    def browse_query_img(self):
        self.filename = tkFileDialog.askopenfile(title='Choose an Image File').name

        # process query image to feature vector
        # load the query image and describe it
        self.query = cv2.imread(self.filename)
        self.queryfeatures = cd.describe(self.query)

        # show query image
        image_file = Image.open(self.filename)
        resized = image_file.resize((100, 100), Image.ANTIALIAS)
        im = ImageTk.PhotoImage(resized)

        if hasattr(self, 'query_img_frame'):
            self.query_img_frame.pack_forget()
            self.result_img_frame.pack_forget()
        self.query_img_frame = Frame(self.master)
        self.query_img_frame.pack()

        image_label = Label(self.query_img_frame, image=im)
        image_label.pack()

        self.query_img_frame.mainloop()


    def show_results_imgs(self):
        if not hasattr(self, 'query'):
            tkMessageBox.showinfo("Error", "Please choose an image")
            return  

        selected_features = self.listbox.curselection()
        if len(selected_features) == 0:
            tkMessageBox.showinfo("Error", "Please select at least one feature")
            return

        if hasattr(self, 'result_img_frame'):
            self.result_img_frame.pack_forget()
        self.result_img_frame = Frame(self.master)
        self.result_img_frame.pack()

        features = [False, False, False, False, False]
        for f in selected_features:
            features[f] = True

        # perform the search and put the results from different features into the list below using append
        results_list = []
        if features[0]: # Color Histogram
            ch_results = self.color_hist_searcher.search(self.queryfeatures)
            results_list.append([ch_results, weights_for_methods[CH]])
        if features[1]: # Visual Keywords
            vw_results = self.bow_searcher.search(self.query)
            results_list.append([vw_results, weights_for_methods[VW]])
        if features[2] and features[3]: # Visual Concept and Deep Learning
            dl_results, vc_results = self.deep_learning_searcher.run_inference_on_image(self.filename)
            results_list.append([vc_results, weights_for_methods[VC]])
            results_list.append([dl_results, weights_for_methods[DL]])
        elif features[2]:  # Visual Concept
            vc_results = self.deep_learning_searcher.run_inference_on_image(self.filename, False, True)[0]
            results_list.append([vc_results, weights_for_methods[VC]])
        elif features[3]:  # Deep Learning
            dl_results = self.deep_learning_searcher.run_inference_on_image(self.filename, True, False)[0]
            results_list.append([dl_results, weights_for_methods[DL]])
                
        results = combine_results(self.image_ids, results_list)
        results = sorted([(v, k) for (k, v) in results.items()])

        self.display_imgs(results, category_names)

    def show_relevant_imgs(self, image_id, results):
        self.result_img_frame.pack_forget()
        self.result_img_frame = Frame(self.master)
        self.result_img_frame.pack()
        
        s = image_id.find("/")
        raw_image_id = image_id[s+1:]

        input_categories = []

        for c in category_names:
            if os.path.isfile(self.search_path + "/" + c + "/" + raw_image_id):
                input_categories.append(c)
 
		self.display_imgs(results, input_categories)

		self.result_img_frame.mainloop()

    def display_imgs(self, results, categories):
    	# show result picturesdp_results
        Label(self.result_img_frame, text="Click a image to get similar images")
        COLUMNS = 5
        image_count = 0
        top_results = get_top_resutls(results, 15, categories)
        for (score, resultID) in top_results:
            # load the result image and display it
            image_count += 1
            r, c = divmod(image_count - 1, COLUMNS)
            im = Image.open( self.search_path + "/" + resultID)
            resized = im.resize((100, 100), Image.ANTIALIAS)
            tkimage = ImageTk.PhotoImage(resized)
            def get_relevance_feedback(i=resultID, r=results):
            	self.show_relevant_imgs(i, r)
            myvar = Button(self.result_img_frame, image=tkimage, command=get_relevance_feedback)
            myvar.image = tkimage
            myvar.grid(row=r, column=c)
            
        self.result_img_frame.mainloop()

root = Tk()
window = UI_class(root,'dataset/train/data')