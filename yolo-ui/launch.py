from userUI import create_interface

if __name__ == "__main__":
    demo = create_interface()
    demo.queue().launch(share=True)