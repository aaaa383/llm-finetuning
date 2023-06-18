import streamlit as st

# UIのセットアップ
st.title('OptiMathPro')

st.markdown("""
私たちはオープンソースLLMの持続的な改善を目標に掲げ、人間とAIが協力して単純な計算問題の解決能力を高めるプロダクトを開発しました。  
具体的には、オープンソースの大規模言語モデル（open-calm-7b)を用いて、ファインチューニングして学習させ、その結果が期待に適わなかった場合には、人間からの正確なフィードバックを受け取り、そのフィードバックを元に再度学習を行います。このプロセスはMLOpsの原則に乗っ取り設計されています。  
このプロダクトを使用することで、初めて問題に取り組んだ際には、小学生レベルの計算が答えられなかったLLMでも、人間からの適切なフィードバックによりその答えを学び、同様の問題に対して将来的に正確な解答ができるようなLLMに成長していくことが期待できます。  
この取り組みは、人間とAIの協力による連続的な学習と改善を可能にします。人間は自分の知識と経験を用いてモデルにフィードバックを提供し、モデルはそのフィードバックを用いて学習を進め、自身の計算能力を向上させていきます。  
このようなプロセスを繰り返すことで、幼児レベルの計算能力から小学3年生レベルの計算能力へと進化することができました。
""")
            

st.markdown("""
**参考論文:**  
LIMA: Less Is More for Alignment   
https://arxiv.org/abs/2305.11206

Enhancing Chat Language Models by Scaling High-quality Instructional Conversations
https://arxiv.org/abs/2305.14233

データセット:  
https://huggingface.co/datasets/saldra/sakura_japanese_dataset
""")