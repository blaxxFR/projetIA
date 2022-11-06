import PySimpleGUI as  ps
import matplotlib.pyplot as plt

class Window:
    def __init__(self):
        self.commun_input = [[ps.Text('Material'), ps.Combo(['Aluminium', 'Copper', 'Steel', 'Titanium'], key='material')],
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
                print(self.get_input(self.tab_group.Get()))


                            
    def start(self):
        self.event_loop(self.window)

        

    def close(self):
        self.window.close()

    
        
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
        """check if the input is valid
        a valid input is a tuple of interger or float"""
        try:
            input1, input2, input3, input4, input5, input6, input7 = self.get_input()
            input1 = float(input1)
            input2 = float(input2)
            input3 = float(input3)
            input4 = float(input4)
            input5 = float(input5)
            return True
        except ValueError:
            return False


if __name__ == '__main__':
    window = Window()
    window.start()


