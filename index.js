const express = require("express");
const bodyParser = require("body-parser");
const Pinecone = require("@pinecone-database/pinecone").Pinecone;
const GoogleGenerativeAIEmbeddings = require("@langchain/google-genai").GoogleGenerativeAIEmbeddings;
const TextLoader = require("langchain/document_loaders/fs/text").TextLoader;
const RecursiveCharacterTextSplitter = require("langchain/text_splitter").RecursiveCharacterTextSplitter;
const TaskType = require("@google/generative-ai").TaskType;
const PineconeStore = require("@langchain/pinecone").PineconeStore;
const PromptTemplate = require("@langchain/core/prompts").PromptTemplate;
const ChatGoogleGenerativeAI = require("@langchain/google-genai").ChatGoogleGenerativeAI;
const StringOutputParser = require("@langchain/core/output_parsers").StringOutputParser;
const cors = require("cors");
const dotenv = require("dotenv");
dotenv.config();
const {
  RunnablePassthrough,
  RunnableSequence,
} = require("@langchain/core/runnables");

const app = express();
app.use(cors());

const port = 5014;
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
  apiKey: '83ce8c5f-183b-4167-87b3-e0b3310daf1d',
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
      apiKey: 'AIzaSyBLuNWqOW6sHrs4Hw0l0B9JhpuzG1NeC4w',
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
  apiKey: 'AIzaSyBLuNWqOW6sHrs4Hw0l0B9JhpuzG1NeC4w',
  modelName: "embedding-001", // 768 dimensions
  taskType: TaskType.RETRIEVAL_DOCUMENT,
});

function combineDocuments(docs) {
  return docs?.map((doc) => doc.pageContent).join("\n\n");
}

async function chat(question, history) {
  const llm = new ChatGoogleGenerativeAI({
    model: "gemini-pro",
    apiKey: 'AIzaSyBLuNWqOW6sHrs4Hw0l0B9JhpuzG1NeC4w',
    maxOutputTokens: 2048,
  });

  const documents = await PineconeStore.fromExistingIndex(embedding_class, {
    pineconeIndex,
  });

  const retriever = documents.asRetriever();

  const standaloneQuestionTemplate = `The given question is related to webX company. create a standalone question
    conversation history: {conv_history}
    question: {question} 
    standalone question:`;
  const standaloneQuestionPrompt = PromptTemplate.fromTemplate(
    standaloneQuestionTemplate
  );

  const answerTemplate = `You are a interactive and helpful supportive bot of WebX company. you are chatting to a customer. so Always be interactive. In interactive way Answer the question from the given information. If you really don't find similar answer of question from information about the company then say "Could you please elaborate more?" And direct the questioner to email webxnepal@gmail.com or whatsApp +9779749761111)
    information: {information}
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
      information: retrieverChain,
      question: ({ original_input }) => original_input.question,
      conv_history: ({ original_input }) => original_input.conv_history,
    },
    answerChain,
  ]);

  const response = await chain.invoke({
    question: question,
    conv_history: [],
  });

  console.log("Question:", question);
  console.log("Response", response);
  return response;
}

// chatBot()

app.get("/", (req, res) => {
  res.send("<h2>ChatBot</h2>");
});

app.post("/chat", async (req, res) => {
  const question = req.body.question;
  const history = req.body.history;

  try {
    const response = await chat(question, history);
    res.status(200).json({ response });
  } catch (err) {
    console.log(err)
    res.status(500).json({
      response:
        "Umm... I request you to contact at webxnepal@gmail.com for more inquiry. ThankYou",
    });
  }
});

app.listen(port, () => {
  console.log(`Server is listening at http://localhost:${port}`);
});
