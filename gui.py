import pickle
import PySimpleGUI as  ps
import matplotlib.pyplot as plt
import numpy as np
from surface import *


def isfloat(num):
    try:
        float(num)
        return True
    except ValueError:
        return False
class Window:
    def __init__(self):
        # Acier de construction': 210E9, 'Acier inxydable': 203E9, 'Aluminum': 69E9, 'Cuivre': 124E9, 'Titane': 114E9, 'Verre': 69E9, 'Béton'
        self.commun_input = [[ps.Text('Material'), ps.Combo(['Acier de construction','Acier inxydable','Aluminum','Cuivre','Titane','Verre','Béton'], key='material')],
                            [ ps.Text('Length [cm]'), ps.InputText(key='L_tot')],
                            [ps.Text('nbr_elements (n)'), ps.InputText(key='nbr_elements')],
                
        ]

        self.rectangle_input = [[ps.Text('Height [cm]'), ps.InputText(key='h_rec')],
                            [ps.Text('Width [cm]'), ps.InputText(key='b_rec')],
        ]

        self.circle_input = [[ps.Text('Radius [cm]'), ps.InputText(key='r_circle')],
        ]

        self.Ipn = [[ps.Text('Height [cm]'), ps.InputText(key='h')],
                            [ps.Text('Base [cm]'), ps.InputText(key='b')],
        ]

        self.rectangle_creuse_input = [[ps.Text('Height_ext [cm]'), ps.InputText(key='h_ext')],
                            [ps.Text('height_int [cm]'), ps.InputText(key='h_int')],
                            [ps.Text('Width_ext [cm]'), ps.InputText(key='b_ext')],
                            [ps.Text('Width_int [cm]'), ps.InputText(key='b_int')],
        ]

        self.circle_creuse_input = [[ps.Text('Radius [cm]'), ps.InputText(key='r_creuse')],
                            [ps.Text('thickness [cm]'), ps.InputText(key='thickness')],
        ]

       # create a layout, user can initialiy choose type of bar, commun input originaly appear, then accoring user choise of bar, show other input

        self.tab_group = ps.TabGroup([[ps.Tab('Rectangle', self.rectangle_input,key='rectangle'), 
                                        
                                                          ps.Tab('Circle', self.circle_input,key='circle'),
                                                          ps.Tab('Rectangle_creuse', self.rectangle_creuse_input,key='rectangle_creuse'),
                                                          ps.Tab('Circle_creuse', self.circle_creuse_input,key='circle_creuse')]], enable_events=True, key='tabgroup')



        # create a layout, user can initialiy choose type of bar, commun input originaly appear, then accoring user choise of bar, show other input
        self.input_column = [
                          [ps.Frame('Commun input', self.commun_input)],
                          [self.tab_group],
                          [ps.Button('Predict'), ps.Button('Exit')],
                          [ps.Multiline(key='output', size=(75, 10))]]

       
        self.forme_column = [[ps.T('Choose what clicking a figure does', enable_events=True)],
           [ps.R('Draw Rectangles', 1, key='-RECT-', enable_events=True)],
           [ps.R('Draw Circle', 1, key='-CIRCLE-', enable_events=True)],
           [ps.R('Draw Line', 1, key='-LINE-', enable_events=True)],
           [ps.R('Draw point', 1,  key='-POINT-', enable_events=True)],
           [ps.R('Erase item', 1, key='-ERASE-', enable_events=True)],
           [ps.R('Erase all', 1, key='-CLEAR-', enable_events=True)],
           [ps.R('Send to back', 1, key='-BACK-', enable_events=True)],
           [ps.R('Bring to front', 1, key='-FRONT-', enable_events=True)],
           [ps.R('Move Everything', 1, key='-MOVEALL-', enable_events=True)],
           [ps.R('Move Stuff', 1, True, key='-MOVE-', enable_events=True)]  ,]

        # load a png 
        self.images_path= [""]

        self.image_colmun = [[ps.Image(filename='schemes_load/Rectangle.png', key='-IMAGE2-')],[ps.Image(filename='schemes_load\Cantilever_beam.png', key='-IMAGE-')]]
        self.canvas_column = [[ps.Graph(canvas_size=(400, 400),
        graph_bottom_left=(0, 0),
        graph_top_right=(400, 400),
        key="-GRAPH-",
        change_submits=True,  # mouse click events
        background_color='white',
        drag_submits=True),ps.Col(self.forme_column)]]
        self.layout = [
            [ ps.Column(self.input_column),ps.VSeparator(), ps.Column(self.image_colmun)]
            
        ]
        
        
       
        """        self.layout = [[ps.Text('Bar type'),self.tab_group],
                        [ps.Button('Predict'), ps.Button('Quit')]
                        + self.commun_input ,
                        [ps.Multiline(key='output', size=(100, 10))]] """


        self.layout_rectangle = self.layout + self.rectangle_input
        self.layout_circle = self.layout + self.circle_input
        self.layout_ipn = self.layout + self.Ipn
        self.layout_rectangle_creuse = self.layout + self.rectangle_creuse_input
        self.layout_circle_creuse = self.layout + self.circle_creuse_input

        self.emat= {'Acier de construction': 210E9, 'Acier inxydable': 203E9, 'Aluminum': 69E9, 'Cuivre': 124E9, 'Titane': 114E9, 'Verre': 69E9, 'Béton': 30E9}
        self.rhomat = {'Acier de construction': 7850, 'Acier inxydable': 7800, 'Aluminum': 2700, 'Cuivre': 8900, 'Titane': 4510, 'Verre': 2500, 'Béton': 2400}

        self.model_rectangle = pickle.load(open('model_rectangle.pkl', 'rb'))
        self.model_rectangle_other_freq = pickle.load(open('model_rectangle_other_freq.pkl', 'rb'))
        self.model_circle = pickle.load(open('model_cercle.pkl', 'rb'))
        self.model_circle_other_freq = pickle.load(open('model_cercle_other_freq.pkl', 'rb'))
        # self.model_ipn = pickle.load(open('model_ipn.pkl', 'rb'))
        self.model_rectangle_creuse = pickle.load(open('model_rectangle_creux.pkl', 'rb'))
        self.model_rectangle_creuse_other_freq = pickle.load(open('model_rectangle_creux_other_freq.pkl', 'rb'))
        self.model_circle_creuse = pickle.load(open('model_cercle_creux.pkl', 'rb'))
        self.model_circle_creuse_other_freq = pickle.load(open('model_cercle_creux_other_freq.pkl', 'rb'))
         # create a window with the layout
   

        #have a white background
        
        

        self.window = ps.Window('Akinapoutre', self.layout,finalize=True)
    """    # self.graph = self.window["-GRAPH-"]  # type: sg.Graph
        self.dragging = False
        self.start_point = self.end_point = self.prior_rect = None
        self.graph.bind('<Button-3>', '+RIGHT+')"""

    # make a function to know wich tab is selected
    def get_tab(self):
        return self.window['tabgroup'].get()


    def event_loop(self,window):
        while True:
            event, values = window.read()
            print(self.get_tab())
            if event == 'Quit' or event == ps.WIN_CLOSED:
                break
            elif event == 'Predict':
                print(self.predict())
            # if rectangle tab is selected update IMAGE2 by rectangle image
            elif self.get_tab() == 'rectangle':


                self.window['-IMAGE2-'].update(filename='schemes_load/Rectangle.png')
            
            # if circle tab is selected update IMAGE2 by circle image
            elif self.get_tab() == 'circle':
                self.window['-IMAGE2-'].update(filename='schemes_load/Cercle.png')
            
            elif self.get_tab() == 'rectangle_creuse':
                self.window['-IMAGE2-'].update(filename='schemes_load/Rectangle_creux.png')
            
            elif self.get_tab() == 'circle_creuse':
                self.window['-IMAGE2-'].update(filename='schemes_load/Cercle_creux.png')

            



            """elif event == "-GRAPH-":  # if there's a "Graph" event, then it's a mouse
                x, y = values["-GRAPH-"]
                if not self.dragging:
                    self.start_point = (x, y)
                    self.dragging = True
                    drag_figures = self.graph.get_figures_at_location((x,y))
                    lastxy = x, y
                else:
                    self.end_point = (x, y)
                if self.prior_rect:
                    self.graph.delete_figure(self.prior_rect)
                delta_x, delta_y = x - lastxy[0], y - lastxy[1]
                lastxy = x,y
                if None not in (self.start_point, self.end_point):
                    if values['-MOVE-']:
                        for fig in drag_figures:
                            self.graph.move_figure(fig, delta_x, delta_y)
                            self.graph.update()
                    elif values['-RECT-']:
                        self.prior_rect = self.graph.draw_rectangle(self.start_point, self.end_point,fill_color='green', line_color='red')
                    elif values['-CIRCLE-']:
                        self.prior_rect = self.graph.draw_circle(self.start_point, self.end_point[0]-self.start_point[0], fill_color='red', line_color='green')
                    elif values['-LINE-']:
                        self.prior_rect = self.graph.draw_line(self.start_point, self.end_point, width=4)
                    elif values['-POINT-']:
                        self.prior_rect = self.graph.draw_point(self.start_point, size=1)
                    elif values['-ERASE-']:
                        for figure in drag_figures:
                            self.graph.delete_figure(figure)
                    elif values['-CLEAR-']:
                        self.graph.erase()
                    elif values['-MOVEALL-']:
                        self.graph.move(delta_x, delta_y)
                    elif values['-FRONT-']:
                        for fig in drag_figures:
                            self.graph.bring_figure_to_front(fig)
                    elif values['-BACK-']:
                        for fig in drag_figures:
                            self.graph.send_figure_to_back(fig)
            elif event.endswith('+UP'):  # The drawing has ended because mouse up
                self.start_point, self.end_point = None, None  # enable grabbing a new rect
                self.dragging = False
                self.prior_rect = None"""


                            
    def start(self):
        self.event_loop(self.window)

        

    def close(self):
        self.window.close()

    def predict(self):

        # 'NbElts', 'S', 'I', 'L', 'E'
        if(self.check_input()):
            var_values = list()
            type_bar = self.tab_group.Get()
            var_values = self.get_input(type_bar)
            if type_bar == 'rectangle':
               
                
                #  ['L_tot','rho', 'h','b']
                #L_tho = L_tot/100
                length = float(var_values[0])/100
                h = float(var_values[1])/100
                b = float(var_values[2])/100
                var_values = [length,float(self.rhomat[var_values [3]]),h,b]
                var_values = np.array(var_values).reshape(1, -1)
                freq1 = self.model_rectangle.predict(var_values)[0][0]
                #aff freq 1 to var_values
                var_values = np.append(var_values,freq1)
                var_values = np.array(var_values).reshape(1, -1)
                other_freq = self.model_rectangle_other_freq.predict(var_values)


                
                tmp_string= "Frequence 1 = " + str(freq1) + " Hz\n"
                # array var value +  self.model_rectangle.predict(var_values)[0][0]
                for i in range(len(other_freq[0])):
                    tmp_string += "Frequence "+ str(i+2) + " = " + str(other_freq[0][i]) + " Hz\n"
                self.window['output'].update(tmp_string)

            if type_bar == 'circle':
                # UPDATE IMAGE2 by circle and update window
                self.window['image2'].update(filename='schemes_loads/cercle.png')
                


                
               
                
                #  ['L_tot','rho', 'r']
                #L_tho = L_tot/100S
                length = float(var_values[0])/100
                r = float(var_values[1])/100
                var_values = (length,float(self.rhomat[var_values [2]]),r)
                var_values = np.array(var_values).reshape(1, -1)
                freq1 = self.model_circle.predict(var_values)[0][0]
                #aff freq 1 to var_values
                var_values = np.append(var_values,freq1)
                var_values = np.array(var_values).reshape(1, -1)
                other_freq = self.model_circle_other_freq.predict(var_values)
                tmp_string= "Frequence 1 = " + str(freq1) + " Hz\n"
                # array var value +  self.model_rectangle.predict(var_values)[0][0]
                for i in range(len(other_freq[0])):
                    tmp_string += "Frequence "+ str(i+2) + " = " + str(other_freq[0][i]) + " Hz\n"
                self.window['output'].update(tmp_string)
            if type_bar == 'ipn':
                surface_ipn = surface_ipn(var_values[1], var_values[2])
            if type_bar == 'rectangle_creuse':
                # update IMAGE2 with schemes_load/rectangle_creuse.png
                
                #  ['L_tot','rho', 'h','b','hr','br']
                #L_tho = L_tot/100
                length = float(var_values[0])/100
                h = float(var_values[1])/100
                b = float(var_values[2])/100
                hr = float(var_values[3])/100
                br = float(var_values[4])/100
                var_values = [length,float(self.rhomat[var_values [5]]),h,b,hr,br]
                var_values = np.array(var_values).reshape(1, -1)
                freq1 = self.model_rectangle_creuse.predict(var_values)[0][0]
                #aff freq 1 to var_values
                var_values = np.append(var_values,freq1)
                var_values = np.array(var_values).reshape(1, -1)
                other_freq = self.model_rectangle_creuse_other_freq.predict(var_values)
                tmp_string= "Frequence 1 = " + str(freq1) + " Hz\n"
                # array var value +  self.model_rectangle.predict(var_values)[0][0]
                for i in range(len(other_freq[0])):
                    tmp_string += "Frequence "+ str(i+2) + " = " + str(other_freq[0][i]) + " Hz\n"
                self.window['output'].update(tmp_string)
            if type_bar == 'circle_creuse':
                # update IMAGE2 with schemes_load/circle_creuse.png
            
                #  ['L_tot','rho', 'r_ext','r_int']
                #L_tho = L_tot/100S
                length = float(var_values[0])/100
                r = float(var_values[1])/100
                rr = float(var_values[2])/100
                var_values = (length,float(self.rhomat[var_values [3]]),r,rr)
                var_values = np.array(var_values).reshape(1, -1)
                freq1 = self.model_circle_creuse.predict(var_values)[0][0]
                #aff freq 1 to var_values
                var_values = np.append(var_values,freq1)
                var_values = np.array(var_values).reshape(1, -1)
                other_freq = self.model_circle_creuse_other_freq.predict(var_values)
                tmp_string= "Frequence 1 = " + str(freq1) + " Hz\n"
                # array var value +  self.model_rectangle.predict(var_values)[0][0]
                for i in range(len(other_freq[0])):
                    tmp_string += "Frequence "+ str(i+2) + " = " + str(other_freq[0][i]) + " Hz\n"
                self.window['output'].update(tmp_string)
        else :
            # popup error and reset input
            ps.popup('Error', 'Please check your input')

        
    def get_input(self,type):
        # return input according to the type of bar
        if type == 'rectangle':
            return self.window['L_tot'].get(), self.window['b_rec'].get(), self.window['h_rec'].get(), self.window['material'].get(), self.window['nbr_elements'].get()
        elif type == 'circle':
            return self.window['L_tot'].get(), self.window['r_circle'].get(), self.window['material'].get(), self.window['nbr_elements'].get()
        elif type == 'ipn':
            return self.window['L_tot'].get(), self.window['h'].get(), self.window['b'].get(), self.window['material'].get(), self.window['nbr_elements'].get()
        elif type == 'rectangle_creuse':
            return self.window['L_tot'].get(), self.window['b_ext'].get(), self.window['h_ext'].get(), self.window['b_int'].get(), self.window['h_int'].get(), self.window['material'].get(), self.window['nbr_elements'].get()
        elif type == 'circle_creuse':
            return self.window['L_tot'].get(), self.window['r_creuse'].get(), self.window['thickness'].get(), self.window['material'].get(), self.window['nbr_elements'].get()
        else:
            return None
    

    def check_input(self):
        print("ALLLER")
        print(self.tab_group.Get())
        # if tab is rectabgle, check if input are correct 
        if self.tab_group.Get() == 'rectangle':
            print("oskour")
            valeurs = self.get_input('rectangle')
            print(valeurs)
            if not(valeurs[0] == '' or valeurs[1] == '' or valeurs[2] == '' or valeurs[3] == '' or valeurs[4] == ''):
                if (valeurs[0].isdigit() or isfloat(valeurs[0])) and (valeurs[1].isdigit() or isfloat(valeurs[1])) and (valeurs[2].isdigit() or isfloat(valeurs[2])) and (valeurs[4].isdigit() or isfloat(valeurs[4])):
                    if(float(valeurs[0]) > 0 and float(valeurs[1]) > 0 and float(valeurs[2]) > 0 and float(valeurs[4]) > 0):
                        return True
            else:
                return False
        elif self.tab_group.Get() == 'circle':
            valeurs = self.get_input('circle')
            if not(valeurs[0] == '' or valeurs[1] == '' or valeurs[2] == '' or valeurs[3] == ''):
                if (valeurs[0].isdigit() or isfloat(valeurs[0])) and (valeurs[1].isdigit() or isfloat(valeurs[1])) and (valeurs[3].isdigit() or isfloat(valeurs[3])):
                    if(float(valeurs[0]) > 0 and float(valeurs[1]) > 0 and float(valeurs[3]) > 0):
                        return True
            else:
                return False
                 
        elif self.tab_group.Get() == 'ipn':
            valeurs = self.get_input('ipn')
            if not(valeurs[0] == '' or valeurs[1] == '' or valeurs[2] == '' or valeurs[3] == '' or valeurs[4] == ''):
                if (valeurs[0].isdigit() or isfloat(valeurs[0])) and (valeurs[1].isdigit() or isfloat(valeurs[1])) and (valeurs[2].isdigit() or isfloat(valeurs[2])) and (valeurs[4].isdigit() or isfloat(valeurs[4])):
                    if(float(valeurs[0]) > 0 and float(valeurs[1]) > 0 and float(valeurs[2]) > 0 and float(valeurs[4]) > 0):
                        return True
            else:
                return False
        elif self.tab_group.Get() == 'rectangle_creuse':
            valeurs = self.get_input('rectangle_creuse')
            if not(valeurs[0] == '' or valeurs[1] == '' or valeurs[2] == '' or valeurs[3] == '' or valeurs[4] == '' or valeurs[5] == '' or valeurs[6] == ''):
                if (valeurs[0].isdigit() or isfloat(valeurs[0])) and (valeurs[1].isdigit() or isfloat(valeurs[1])) and (valeurs[2].isdigit() or isfloat(valeurs[2])) and (valeurs[3].isdigit() or isfloat(valeurs[3])) and (valeurs[4].isdigit() or isfloat(valeurs[4])) and (valeurs[6].isdigit() or isfloat(valeurs[6])):
                    if(float(valeurs[0]) > 0 and float(valeurs[1]) > 0 and float(valeurs[2]) > 0 and float(valeurs[3]) > 0 and float(valeurs[4]) > 0 and float(valeurs[6]) > 0):
                        return True
            else:
                return False
        elif self.tab_group.Get() == 'circle_creuse':
            valeurs = self.get_input('circle_creuse')
            if not(valeurs[0] == '' or valeurs[1] == '' or valeurs[2] == '' or valeurs[3] == '' or valeurs[4] == ''):
                if (valeurs[0].isdigit() or isfloat(valeurs[0])) and (valeurs[1].isdigit() or isfloat(valeurs[1])) and (valeurs[2].isdigit() or isfloat(valeurs[2])) and (valeurs[4].isdigit() or isfloat(valeurs[4])):
                    if(float(valeurs[0]) > 0 and float(valeurs[1]) > 0 and float(valeurs[2]) > 0 and float(valeurs[4]) > 0):
                        return True
            else:
                return False
        else:
            return False


            
if __name__ == '__main__':
    window = Window()
    window.start()


