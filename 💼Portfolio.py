# import the Streamlit library
import streamlit as st
from streamlit_option_menu import option_menu
from utils.constants import *

# configure page settings
st.set_page_config(page_title='Template' ,layout="wide",initial_sidebar_state="auto", page_icon='👧🏻') # always show the sidebar

# load local CSS styles
def local_css(file_name):
    with open(file_name) as f:
        st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)
        
local_css("styles/styles_main.css")

# get the variables from constants.py
pronoun = info['Pronoun']
full_name = info['Full_Name']

# app sidebar
with st.sidebar:
    st.markdown("""
                # Chat with my AI assistant
                """)
    with st.expander("Click here to see FAQs"):
        st.info(
            """
            - What are {pronoun} strengths and weaknesses?
            - What is {pronoun} expected salary?
            - What is {pronoun} latest project?
            - When can {pronoun} start to work?
            - Tell me about {pronoun} professional background
            - What is {pronoun} skillset?
            - What is {pronoun} contact?
            - What are {pronoun} achievements?
            """
        )
        
    st.caption("© Made by {full_name} 2023. All rights reserved.")

import requests



def hero(content1, content2):
    st.markdown(f'<h1 style="text-align:center;font-size:60px;border-radius:2%;">'
                f'<span>{content1}</span><br>'
                f'<span style="color:black;font-size:22px;">{content2}</span></h1>', 
                unsafe_allow_html=True)

with st.container():
    col1,col2 = st.columns([8,3])

with col1:
    hero("Hi, I'm Vicky Kuo👋", "A Tech Educator and AI Enthusiast at cognitiveclass.ai")
    st.write("")
    st.write(info['About'])

    from streamlit_extras.switch_page_button import switch_page
    col_1, col_2, temp = st.columns([0.35,0.2,0.45])
    with col_1:
        btn1 = st.button("Chat with My AI Assistant")
        if btn1:
            switch_page("AI_Assistant_Chat")
    with col_2:
        btn2 = st.button("My Resume")
        if btn2:
            switch_page("Resume")

import streamlit.components.v1 as components

def change_button_color(widget_label, background_color='transparent'):
    htmlstr = f"""
        <script>
            var elements = window.parent.document.querySelectorAll('button');
            for (var i = 0; i < elements.length; ++i) {{ 
                if (elements[i].innerText == '{widget_label}') {{ 
                    elements[i].style.background = '{background_color}'
                }}
            }}
        </script>
        """
    components.html(f"{htmlstr}", height=0, width=0)

change_button_color('Chat with My AI Assistant', '#0cc789') 

from  PIL import Image
with col2:
    profile = Image.open("images/profile.png")
    st.image(profile, width=280)

            
with st.container():
    st.write("---")
    st.subheader('🚀 Project Showcase')

    projects = [
        {
            "title": "Unlocking Multilingual Magic: Babel Fish with LLM STT TTS",
            "description": "Text-to-Speech, Speech-To-Text, LLM, watsonx",
            "image_url": "https://sn-assets.s3.us.cloud-object-storage.appdomain.cloud/projects/babel.png",
            "link": "https://cognitiveclass.ai/courses/course-v1:IBMSkillsNetwork+GPXX0PPIEN+v1"
        },
        {
            "title": "Summarize Your Private Data with LLMs & Generative AI",
            "description": "LLM, watsonx, watsonx.ai",
            "image_url": "https://sn-assets.s3.us.cloud-object-storage.appdomain.cloud/summary.png",
            "link": "https://cognitiveclass.ai/courses/course-v1:IBMSkillsNetwork+GPXX0DTPEN+v1"
        },      
        {
            "title": "Improve Customer Support Efficiency with Open-Source LLM",
            "description": "Generative AI",
            "image_url": "https://sn-assets.s3.us.cloud-object-storage.appdomain.cloud/projects/CS.png",
            "link": "https://cognitiveclass.ai/courses/course-v1:IBMSkillsNetwork+GPXX0FUSEN+v1"
        },
        {
            "title": "A Quick Introduction to Machine Learning",
            "description": "Machine Learning",
            "image_url": "https://sn-assets.s3.us.cloud-object-storage.appdomain.cloud/projects/ML.png",
            "link": "https://cognitiveclass.ai/courses/course-v1:IBMSkillsNetwork+ML0104EN+v1"
        },
        {
            "title": "Reinforcement Learning and Deep Learning Essentials",
            "description": "Reinforcement Learning, Keras, PyTorch",
            "image_url": "https://sn-assets.s3.us.cloud-object-storage.appdomain.cloud/projects/RL.png",
            "link": "https://cognitiveclass.ai/courses/course-v1:IBMSkillsNetwork+ML0105EN+v1"
        },
        {
            "title": "Automate ML Pipelines Using Apache Airflow",
            "description": "Machine Learning Pipeline",
            "image_url": "https://sn-assets.s3.us.cloud-object-storage.appdomain.cloud/projects/airflow.png",
            "link": "https://cognitiveclass.ai/courses/course-v1:IBM+GPXX0DNQEN+v1"
        },
        {
            "title": "Create Your Own ChatGPT-like Website with Open Source LLMs",
            "description": "LLM, Chatbot, Generative AI",
            "image_url": "https://sn-assets.s3.us.cloud-object-storage.appdomain.cloud/projects/chatgpt.png",
            "link": "https://cognitiveclass.ai/courses/course-v1:IBMSkillsNetwork+GPXX04ESEN+v1"
        },
        {
            "title": "A Quick Introduction to Machine Learning",
            "description": "Machine Learning",
            "image_url": "https://sn-assets.s3.us.cloud-object-storage.appdomain.cloud/projects/ML.png",
            "link": "https://cognitiveclass.ai/courses/course-v1:IBMSkillsNetwork+ML0104EN+v1"
        },
        {
            "title": "Build an AI Web App for Diamond Price Prediction",
            "description": "Machine Learning, Data Science",
            "image_url": "https://sn-assets.s3.us.cloud-object-storage.appdomain.cloud/projects/diamond.png",
            "link": "https://cognitiveclass.ai/courses/course-v1:IBMSkillsNetwork+GPXX0FTNEN+v1"
        }                  
    ]

    def display_project(col, project):
        with col:
            st.markdown(
                f'<a href="{project["link"]}" target="_blank" class="portfolio-item" data-id="3">'
                f'<img src="{project["image_url"]}" style="width:100%;height:auto;"></a>',
                unsafe_allow_html=True,
            )
            st.markdown(f'<p style="font-size: 16px; font-weight: bold;">{project["title"]}</p>', unsafe_allow_html=True)
            st.markdown(f'<p style="font-size: 14px">{project["description"]}</p>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1,1,1])
    columns = [col1, col2, col3]

    # Adjust the range(0, len(projects), 3) accordingly if your projects length is not a multiple of 3
    for i in range(0, len(projects), 3):
        for j, col in enumerate(columns):
            if i + j < len(projects):  # Check if project index is within range
                display_project(col, projects[i + j])
st.markdown(""" <a href={}> <em>👀 Click here to see more </a>""".format(info['Project']), unsafe_allow_html=True)
    


import streamlit.components.v1 as components
    
with st.container():
    st.markdown("""""")
    st.subheader('✍️ Medium')
    page = requests.get(info['Medium'])
    col1,col2 = st.columns([0.95, 0.05])
    components.html(embed_rss['rss'],height=300)
    st.markdown(""" <a href={}> <em>👀 Click here to see more</a>""".format(info['Medium']), unsafe_allow_html=True)

st.write("---")
with st.container():  
    col1,col2,col3 = st.columns([0.475, 0.475, 0.05])
        
    with col1:
        st.subheader("👄 Coworker Endorsements")
        components.html(
        """
        <!DOCTYPE html>
        <html>
        <head>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            * {box-sizing: border-box;}
            .mySlides {display: none;}
            img {vertical-align: middle;}

            /* Slideshow container */
            .slideshow-container {
            position: relative;
            margin: auto;
            width: 100%;
            }

            /* The dots/bullets/indicators */
            .dot {
            height: 15px;
            width: 15px;
            margin: 0 2px;
            background-color: #eaeaea;
            border-radius: 50%;
            display: inline-block;
            transition: background-color 0.6s ease;
            }

            .active {
            background-color: #6F6F6F;
            }

            /* Fading animation */
            .fade {
            animation-name: fade;
            animation-duration: 1s;
            }

            @keyframes fade {
            from {opacity: .4} 
            to {opacity: 1}
            }

            /* On smaller screens, decrease text size */
            @media only screen and (max-width: 300px) {
            .text {font-size: 11px}
            }
            </style>
        </head>
        <body>
            <div class="slideshow-container">
                <div class="mySlides fade">
                <img src="https://user-images.githubusercontent.com/90204593/238169843-12872392-f2f1-40a6-a353-c06a2fa602c5.png" style="width:100%">
                </div>

                <div class="mySlides fade">
                <img src="https://user-images.githubusercontent.com/90204593/238171251-5f4c5597-84d4-4b4b-803c-afe74e739070.png" style="width:100%">
                </div>

                <div class="mySlides fade">
                <img src="https://user-images.githubusercontent.com/90204593/238171242-53f7ceb3-1a71-4726-a7f5-67721419fef8.png" style="width:100%">
                </div>

            </div>
            <br>

            <div style="text-align:center">
                <span class="dot"></span> 
                <span class="dot"></span> 
                <span class="dot"></span> 
            </div>

            <script>
            let slideIndex = 0;
            showSlides();

            function showSlides() {
            let i;
            let slides = document.getElementsByClassName("mySlides");
            let dots = document.getElementsByClassName("dot");
            for (i = 0; i < slides.length; i++) {
                slides[i].style.display = "none";  
            }
            slideIndex++;
            if (slideIndex > slides.length) {slideIndex = 1}    
            for (i = 0; i < dots.length; i++) {
                dots[i].className = dots[i].className.replace("active", "");
            }
            slides[slideIndex-1].style.display = "block";  
            dots[slideIndex-1].className += " active";
            }

            var interval = setInterval(showSlides, 2500); // Change image every 2.5 seconds

            function pauseSlides(event)
            {
                clearInterval(interval); // Clear the interval we set earlier
            }
            function resumeSlides(event)
            {
                interval = setInterval(showSlides, 2500);
            }
            // Set up event listeners for the mySlides
            var mySlides = document.getElementsByClassName("mySlides");
            for (i = 0; i < mySlides.length; i++) {
            mySlides[i].onmouseover = pauseSlides;
            mySlides[i].onmouseout = resumeSlides;
            }
            </script>

            </body>
            </html> 

            """,
                height=270,
    )

    with col2:
        st.subheader("📨 Contact Me")

        contact_form = """
        <form action="https://formsubmit.co/vicky.kuo.contact@gmail.com" method="POST">
            <input type="hidden" name="_captcha value="false">
            <input type="text" name="name" placeholder="Your name" required>
            <input type="email" name="email" placeholder="Your email" required>
            <textarea name="message" placeholder="Your message here" required></textarea>
            <button type="submit">Send</button>
        </form>
        """
        st.markdown(contact_form, unsafe_allow_html=True)
