# import the necessary packages
from params import *
from pyimagesearch.color_hist_searcher import *
from pyimagesearch.bow_searcher import *
from pyimagesearch.deep_learning_searcher import *
from Tkinter import *
import tkFileDialog
import tkMessageBox
from PIL import Image, ImageTk

def combine_results(image_ids, results):
    if len(results) > 1:
        final_result = {}
        for image_id in image_ids:
            score = reduce(lambda x, y: x + y, (r[image_id] for r in results))
            final_result[image_id] = score
        return final_result
    else:
        return results[0]

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
            "Text"
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
            results_list.append(ch_results)
        if features[1]: # Visual Keywords
            vw_results = self.bow_searcher.search(self.query)
            results_list.append(vw_results)
        if features[2] and features[3]: # Visual Concept and Deep Learning
            dp_results, vc_results = self.deep_learning_searcher.run_inference_on_image(self.filename)
            results_list.append(vc_results)
            results_list.append(dp_results)
        elif features[2] or features[3]: # Visual Concept or Deep Learning
            results_list.append(self.deep_learning_searcher.run_inference_on_image(self.filename, features[3], features[2])[0])
                
        results = combine_results(self.image_ids, results_list)
        results = sorted([(v, k) for (k, v) in results.items()])

        # show result picturesdp_results
        COLUMNS = 5
        image_count = 0
        for (score, resultID) in results[:20]:
            # load the result image and display it
            image_count += 1
            r, c = divmod(image_count - 1, COLUMNS)
            im = Image.open( self.search_path + "/" + resultID)
            resized = im.resize((100, 100), Image.ANTIALIAS)
            tkimage = ImageTk.PhotoImage(resized)
            myvar = Label(self.result_img_frame, image=tkimage)
            myvar.image = tkimage
            myvar.grid(row=r, column=c)

        self.result_img_frame.mainloop()


root = Tk()
window = UI_class(root,'dataset/train/data')