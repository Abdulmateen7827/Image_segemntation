import sys
from src.logger import logging

def get_error_message(error, error_detail:sys):
    _,_,exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = f"Error occured in script {file_name}, line no {exc_tb.tb_frame.f_lineno}, error message{str(error)}"

    return error_message

class CustomException(Exception):
    def __init__(self,error_message,error_detail:sys):
        #Inheriting from the Exception class 
        super().__init__(error_message)
        self.error_message = get_error_message(error_message,error_detail=error_detail)

    def __str__(self) -> str:
        return self.error_message


        

