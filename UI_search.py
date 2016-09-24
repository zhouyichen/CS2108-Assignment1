# import the necessary packages
from params import *
from pyimagesearch.color_hist_searcher import *
from pyimagesearch.bow_searcher import *
from pyimagesearch.deep_learning_searcher import *
from Tkinter import *
import tkFileDialog
from PIL import Image, ImageTk

def combine_results(image_ids, results):
    if len(results) > 1:
        final_result = {}
        for image_id in image_ids:
            score = reduce(lambda x, y: x*y, (r[image_id] for r in results))
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
        self.cbutton = Button(topframe, text=" Search ", command=self.show_results_imgs)
        self.cbutton.grid(row=1, column=2)
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
        self.result_img_frame = Frame(self.master)
        self.result_img_frame.pack()

        # perform the search
        
        ch_results = self.color_hist_searcher.search(self.queryfeatures)
        vw_results = self.bow_searcher.search(self.query)
        dp_results, vc_results = self.deep_learning_searcher.run_inference_on_image(self.filename)

        '''put the results from different features into the list below using append'''
        results_list = []
        results_list.append(dp_results)

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