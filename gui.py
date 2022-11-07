import PySimpleGUI as  ps
import matplotlib.pyplot as plt
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
        self.commun_input = [[ps.Text('Material'), ps.Combo(['Acier de construction','Acier inxydable','Aluminum','Cuibre','Titane','Verre','Béton'], key='material')],
                            [ps.Text('Length'), ps.InputText(key='L_tot')],
                            [ps.Text('nbr_elements'), ps.InputText(key='nbr_elements')],
                
        ]

        self.rectangle_input = [[ps.Text('Height'), ps.InputText(key='h_rec')],
                            [ps.Text('Width'), ps.InputText(key='b_rec')],
        ]

        self.circle_input = [[ps.Text('Radius'), ps.InputText(key='r_circle')],
        ]

        self.Ipn = [[ps.Text('Height'), ps.InputText(key='h')],
                            [ps.Text('Base'), ps.InputText(key='b')],
        ]

        self.rectangle_creuse_input = [[ps.Text('Height_ext'), ps.InputText(key='h_ext')],
                            [ps.Text('height_int'), ps.InputText(key='h_int')],
                            [ps.Text('Width_ext'), ps.InputText(key='b_ext')],
                            [ps.Text('Width_int'), ps.InputText(key='b_int')],
        ]

        self.circle_creuse_input = [[ps.Text('Radius'), ps.InputText(key='r_creuse')],
                            [ps.Text('thickness'), ps.InputText(key='thickness')],
        ]

       # create a layout, user can initialiy choose type of bar, commun input originaly appear, then accoring user choise of bar, show other input

        self.tab_group = ps.TabGroup([[ps.Tab('Rectangle', self.rectangle_input,key='rectangle'), 
                                                          ps.Tab('ipn', self.Ipn,key='ipn'),
                                                          ps.Tab('Circle', self.circle_input,key='circle'),
                                                          ps.Tab('Rectangle_creuse', self.rectangle_creuse_input,key='rectangle_creuse'),
                                                          ps.Tab('Circle_creuse', self.circle_creuse_input,key='circle_creuse')]], enable_events=True, key='tabgroup')


        self.layout = [[ps.Text('Bar type'),self.tab_group],
                        [ps.Button('Predict'), ps.Button('Quit')]
                        + self.commun_input ,
                        [ps.Multiline(key='output', size=(100, 10))]] 


        self.layout_rectangle = self.layout + self.rectangle_input
        self.layout_circle = self.layout + self.circle_input
        self.layout_ipn = self.layout + self.Ipn
        self.layout_rectangle_creuse = self.layout + self.rectangle_creuse_input
        self.layout_circle_creuse = self.layout + self.circle_creuse_input

        self.emat= {'Acier de construction': 210E9, 'Acier inxydable': 203E9, 'Aluminum': 69E9, 'Cuivre': 124E9, 'Titane': 114E9, 'Verre': 69E9, 'Béton': 30E9}
        self.rhomat = {'Acier de construction': 7850, 'Acier inxydable': 7800, 'Aluminum': 2700, 'Cuivre': 8900, 'Titane': 4510, 'Verre': 2500, 'Béton': 2400}

         # create a window with the layout
   
        self.window = ps.Window('Bar', self.layout)
       
    def selected_tab(self):
        """return the selected tab"""
        return self.tab_group.find_key_from_tab_name()
        

    def event_loop(self,window):
        while True:
            event, values = window.read()
            if event == 'Quit' or event == ps.WIN_CLOSED:
                break
            elif event == 'Predict':
                print(self.predict())


                            
    def start(self):
        self.event_loop(self.window)

        

    def close(self):
        self.window.close()

    def predict(self):

        # 'NbElts', 'S', 'I', 'L', 'E'
        if(self.check_input()):
            var_values = tuple()
            type_bar = self.tab_group.Get()
            var_values = self.get_input(type_bar)
            if type_bar == 'rectangle':
                nb_elts = var_values[4]
                surface_rec = surface_rectangle(int(var_values[1]), int(var_values[2]))
                l = int(var_values[0])/int(var_values[4])
                emat = self.emat[var_values[3]]
                I = (int(var_values[1])*int(var_values[2])**3)/12
                return nb_elts, surface_rec, I, l, emat

            if type_bar == 'circle':
                surface_circle = surface_circle(var_values[1])
            if type_bar == 'ipn':
                surface_ipn = surface_ipn(var_values[1], var_values[2])
            if type_bar == 'rectangle_creuse':
                surface_rectangle_creuse = surface_rectangle_creuse(var_values[1], var_values[2], var_values[3], var_values[4])
            if type_bar == 'circle_creuse':
                surface_circle_creuse = surface_circle_creuse(var_values[1], var_values[2])
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
                    if(valeurs[0] > 0 and valeurs[1] > 0 and valeurs[2] > 0 and valeurs[4] > 0):
                        return True
            else:
                return False
        elif self.tab_group.Get() == 'circle':
            valeurs = self.get_input('circle')
            if not(valeurs[0] == '' or valeurs[1] == '' or valeurs[2] == '' or valeurs[3] == ''):
                if (valeurs[0].isdigit() or isfloat(valeurs[0])) and (valeurs[1].isdigit() or isfloat(valeurs[1])) and (valeurs[3].isdigit() or isfloat(valeurs[3])):
                    if(valeurs[0] > 0 and valeurs[1] > 0 and valeurs[3] > 0):
                        return True
            else:
                return False
                 
        elif self.tab_group.Get() == 'ipn':
            valeurs = self.get_input('ipn')
            if not(valeurs[0] == '' or valeurs[1] == '' or valeurs[2] == '' or valeurs[3] == '' or valeurs[4] == ''):
                if (valeurs[0].isdigit() or isfloat(valeurs[0])) and (valeurs[1].isdigit() or isfloat(valeurs[1])) and (valeurs[2].isdigit() or isfloat(valeurs[2])) and (valeurs[4].isdigit() or isfloat(valeurs[4])):
                    if(valeurs[0] > 0 and valeurs[1] > 0 and valeurs[2] > 0 and valeurs[4] > 0):
                        return True
            else:
                return False
        elif self.tab_group.Get() == 'rectangle_creuse':
            valeurs = self.get_input('rectangle_creuse')
            if not(valeurs[0] == '' or valeurs[1] == '' or valeurs[2] == '' or valeurs[3] == '' or valeurs[4] == '' or valeurs[5] == '' or valeurs[6] == ''):
                if (valeurs[0].isdigit() or isfloat(valeurs[0])) and (valeurs[1].isdigit() or isfloat(valeurs[1])) and (valeurs[2].isdigit() or isfloat(valeurs[2])) and (valeurs[3].isdigit() or isfloat(valeurs[3])) and (valeurs[4].isdigit() or isfloat(valeurs[4])) and (valeurs[6].isdigit() or isfloat(valeurs[6])):
                    if(valeurs[0] > 0 and valeurs[1] > 0 and valeurs[2] > 0 and valeurs[3] > 0 and valeurs[4] > 0 and valeurs[6] > 0):
                        return True
            else:
                return False
        elif self.tab_group.Get() == 'circle_creuse':
            valeurs = self.get_input('circle_creuse')
            if not(valeurs[0] == '' or valeurs[1] == '' or valeurs[2] == '' or valeurs[3] == '' or valeurs[4] == ''):
                if (valeurs[0].isdigit() or isfloat(valeurs[0])) and (valeurs[1].isdigit() or isfloat(valeurs[1])) and (valeurs[2].isdigit() or isfloat(valeurs[2])) and (valeurs[4].isdigit() or isfloat(valeurs[4])):
                    if(valeurs[0] > 0 and valeurs[1] > 0 and valeurs[2] > 0 and valeurs[4] > 0):
                        return True
            else:
                return False
        else:
            return False
            

            
if __name__ == '__main__':
    window = Window()
    window.start()


