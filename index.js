import express from 'express';
import bodyParser from 'body-parser';
import { Pinecone } from "@pinecone-database/pinecone";
import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
import { TextLoader } from "langchain/document_loaders/fs/text";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { TaskType } from "@google/generative-ai";
import { PineconeStore } from "@langchain/pinecone";
import { PromptTemplate } from "@langchain/core/prompts";
import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { StringOutputParser } from "@langchain/core/output_parsers";
import cors from 'cors'
import dotenv from 'dotenv';
dotenv.config();
import {
  RunnablePassthrough,
  RunnableSequence,
} from "@langchain/core/runnables";

const app = express();
app.use(cors());

const port = 3000 || process.env.PORT
app.use(bodyParser.json());
function formatConvHistory(messages) {
  return messages
    ?.map((message, i) => {
      if (i % 2 === 0) {
        return `Human: ${message}`;
      } else {
        return `AI: ${message}`;
      }
    })
    .join("\n");
}
const pinecone = new Pinecone({
  apiKey: process.env.PINECONE_API_KEY,
});
const pineconeIndex = pinecone.Index("document");
async function chatBot() {
  const loader = new TextLoader("./data/personal.txt");
  const textLoader = await loader.load();
  console.log(textLoader);
  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 100,
    chunkOverlap: 50,
    separator: ["\n\n", "\n", " ", "-", "."], // Choose a primary separator
  });
  const output = await splitter.createDocuments([textLoader[0].pageContent]);
  console.log("output:", output);
  await PineconeStore.fromDocuments(
    output,
    new GoogleGenerativeAIEmbeddings({
      apiKey: process.env.GEMI_API_KEY,
      modelName: "embedding-001", // 768 dimensions
      taskType: TaskType.RETRIEVAL_DOCUMENT,
    }),
    {
      pineconeIndex,
    }
  );
}
// chatBot();
const index_name = "document";
const embedding_class = new GoogleGenerativeAIEmbeddings({
  apiKey:process.env.GEMI_API_KEY,
  modelName: "embedding-001", // 768 dimensions
  taskType: TaskType.RETRIEVAL_DOCUMENT,
});
function combineDocuments(docs){
    return docs?.map((doc)=>doc.pageContent).join('\n\n')
}
async function chat(question) {
  const llm = new ChatGoogleGenerativeAI({
    model: "gemini-pro",
    apiKey: process.env.GEMI_API_KEY,
    maxOutputTokens: 2048,
  });
  const documents = await PineconeStore.fromExistingIndex(
      embedding_class,
      {pineconeIndex},
  );
  const retriever = documents.asRetriever()
  const standaloneQuestionTemplate = `From the given question .Create a standalone question ,
    conversation history: {conv_history}
    question: {question} 
    standalone question:`;
  const standaloneQuestionPrompt = PromptTemplate.fromTemplate(
    standaloneQuestionTemplate
  );
  const answerTemplate = `You are a interactive and helpful bot of WebX company .. you are chatting to a customer. so Always be interactive .Find the answer from the given context. If you really don't find answer from context about the company, say "I'm sorry, I don't know the answer to that." And direct the questioner to email webxnepal@gmail.com.
    context: {context}
    conversation history: {conv_history}
    question: {question}
    answer: `;
  const answerPrompt = PromptTemplate.fromTemplate(answerTemplate);
  const standaloneQuestionChain = standaloneQuestionPrompt
    .pipe(llm)
    .pipe(new StringOutputParser());

  const retrieverChain = RunnableSequence.from([
    (prevResult) => prevResult.standalone_question,
    retriever,
    combineDocuments,
  ]);

  const answerChain = answerPrompt.pipe(llm).pipe(new StringOutputParser());

  const chain = RunnableSequence.from([
    {
      standalone_question: standaloneQuestionChain,
      original_input: new RunnablePassthrough(),
    },
    {
      context: retrieverChain,
      question: ({ original_input }) => original_input.question,
      conv_history: ({ original_input }) => original_input.conv_history,
    },
    answerChain,
  ]);
  const response = await chain.invoke({
    question: question,
    conv_history: formatConvHistory([]),
  });
  console.log(response)
  return response;
}
// chatBot()
app.get("/",(req, res) => {
  res.send("<h2>ChatBot</h2>")
})
app.post('/chat', async (req, res) => {
  const question = req.body.question;
  try{
  const response = await chat(question);
  res.json({ response });
}catch(err){
  res.json({ response:"Umm... I request you to contact at webxnepal@gmail.com for more inquiry. ThankYou" })
}
});
app.listen(port, () => {
  console.log(`Server is listening at http://localhost:${port}`);
});