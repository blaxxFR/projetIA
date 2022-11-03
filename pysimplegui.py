import PySimpleGUI


class Window:
    def __init__(self):
        self.layout = [[PySimpleGUI.Text('L_tot'), PySimpleGUI.InputText(key='L_tot')],
                       [PySimpleGUI.Text('rho'), PySimpleGUI.InputText(key='rho')],
                       [PySimpleGUI.Text('h'), PySimpleGUI.InputText(key='h')],
                       [PySimpleGUI.Text('b'), PySimpleGUI.InputText(key='b')],
                       [PySimpleGUI.Text('Material'), PySimpleGUI.Combo(['Aluminium', 'Copper', 'Steel', 'Titanium'], key='material')],
                       [PySimpleGUI.Button('Submit'), PySimpleGUI.Button('Quit')]]
        self.window = PySimpleGUI.Window('Input', self.layout)

    def start(self):
        while True:
            event, values = self.window.read()
            # draws dynamically according to the values of the input a rectangle of the right size
            self.draw_rectangle()
            if event == 'Submit':
                if self.check_input():
                    self.close()
                    break
                else:
                    PySimpleGUI.popup('Invalid input')

            if event == 'Quit' or event == PySimpleGUI.WIN_CLOSED:
                break
            elif event == 'Submit':
                print(values)
                self.window.close()
                return values

    def close(self):
        self.window.close()
    
    def get_input(self):
        return self.window['L_tot'].get(), self.window['rho'].get(), self.window['h'].get(), self.window['b'].get(), self.window['material'].get()

    def check_input(self):
        """check if the input is valid
        a valid input is a tuple of interger or float"""
        try:
            input1, input2, input3, input4, input5 = self.get_input()
            input1 = float(input1)
            input2 = float(input2)
            input3 = float(input3)
            input4 = float(input4)
            return True
        except ValueError:
            return False

    
    def draw_rectangle(self):
        """

        
        """
        pass
        




if __name__ == '__main__':
    window = Window()
    window.start()
    window.close()