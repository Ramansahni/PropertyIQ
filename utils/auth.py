import streamlit as st
import random

def generate_captcha():
    num1 = random.randint(1, 10)
    num2 = random.randint(1, 10)
    return num1, num2, num1 + num2

def display_captcha(key_prefix):
    if f'{key_prefix}_captcha_ans' not in st.session_state:
        num1, num2, ans = generate_captcha()
        st.session_state[f'{key_prefix}_captcha_q'] = f"What is {num1} + {num2}?"
        st.session_state[f'{key_prefix}_captcha_ans'] = ans
    
    user_ans = st.text_input(st.session_state[f'{key_prefix}_captcha_q'], key=f'{key_prefix}_captcha_input')
    
    if user_ans:
        try:
            return int(user_ans) == st.session_state[f'{key_prefix}_captcha_ans']
        except ValueError:
            return False
    return False

def reset_captcha(key_prefix):
    if f'{key_prefix}_captcha_ans' in st.session_state:
        del st.session_state[f'{key_prefix}_captcha_ans']
    if f'{key_prefix}_captcha_q' in st.session_state:
        del st.session_state[f'{key_prefix}_captcha_q']
