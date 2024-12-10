import { useState, useRef, useEffect } from "react";
import "./App.css";
import axios from "axios";
import ReactMarkdown from "react-markdown";

import {
  GoogleGenerativeAI,
  HarmCategory,
  HarmBlockThreshold,
} from "@google/generative-ai";

const apiKey = "GEMINI_API_KEY";
const genAI = new GoogleGenerativeAI(apiKey);

const model = genAI.getGenerativeModel({
  model: "gemini-1.5-pro",
});

const generationConfig = {
  temperature: 1,
  topP: 0.95,
  topK: 40,
  maxOutputTokens: 8192,
  responseMimeType: "text/plain",
};

function App() {
  const [chatHistory, setChatHistory] = useState([]);
  const [question, setQuestion] = useState("");
  const [generatingAnswer, setGeneratingAnswer] = useState(false);
  const [file, setFile] = useState(null); // To hold uploaded image
  const chatContainerRef = useRef(null);

  const [messages, setMessages] = useState([]);

  const AnswerFromGemini = async (userResponse) => {
    try {
      const userInput = userResponse;

      setMessages([...messages, { role: "user", content: userInput }]);

      const response = await fetch("http://127.0.0.1:5000/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ message: userInput }),
      });

      const result = await response.json();
      console.log(result);

      setChatHistory((prev) => [
        ...prev,
        { type: "answer", content: result.response },
      ]);
    } catch (error) {
      console.error("Error fetching fracture details:", error);
      return "Sorry, I couldn't retrieve the details. Please try again later.";
    }
  };

  const fetchFractureDetailsFromGemini = async (bodyPart) => {
    try {
      const userInput = `tell me more about the fracture in my ${bodyPart}`;

      const response = await fetch("http://127.0.0.1:5000/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ message: userInput }),
      });

      const result = await response.json();
 
      setChatHistory((prev) => [
        ...prev,
        { type: "answer", content: result.response },
      ]);
    } catch (error) {
      console.error("Error fetching fracture details:", error);
      return "Sorry, I couldn't retrieve the details. Please try again later.";
    }
  };

  useEffect(() => {
    if (chatContainerRef.current) {
      chatContainerRef.current.scrollTop =
        chatContainerRef.current.scrollHeight;
    }
  }, [chatHistory, generatingAnswer]);

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
  };

  async function generateAnswer(e) {
    e.preventDefault();

    if (!file) {
      alert("Please upload an image first.");
      return;
    }

    setGeneratingAnswer(true);
    const formData = new FormData();
    formData.append("file", file);

    setChatHistory((prev) => [
      ...prev,
      { type: "question", content: "Analyzing the X-ray image..." },
    ]);

    try {
      const response = await axios.post(
        "http://127.0.0.1:5000/predict",
        formData,
        {
          headers: {
            "Content-Type": "multipart/form-data",
          },
        }
      );

      const res = response.data;

      setChatHistory((prev) => [
        ...prev,
        { type: "answer", content: `Prediction: ${res.prediction}` },
        { type: "answer", content: `Body part affected: ${res.body_part}` },
        {
          type: "answer",
          content:
            "Would you like more information about the fracture in the body part?",
        },
      ]);
    } catch (error) {
      console.error(error);
      setChatHistory((prev) => [
        ...prev,
        {
          type: "answer",
          content: "Sorry, there was an issue processing the image.",
        },
      ]);
    }

    setGeneratingAnswer(false);
  }

  async function handleFollowUpResponse(e) {
    e.preventDefault();
    const userResponse = question.trim().toLowerCase();
    setChatHistory((prev) => [
      ...prev,
      { type: "question", content: userResponse },
    ]);

    setQuestion(""); 
    if (userResponse === "yes") {
      const bodyPart =
        chatHistory[chatHistory.length - 2]?.content?.split(": ")[1] ||
        "unknown body part";

      await fetchFractureDetailsFromGemini(bodyPart);
    } else if (userResponse === "no") {
      setChatHistory((prev) => [
        ...prev,
        {
          type: "answer",
          content: "If you have any other questions, feel free to ask!",
        },
      ]);
    } else {
      await AnswerFromGemini(userResponse);
    }

    
  }

  return (
    <div className="fixed inset-0 bg-gradient-to-r from-blue-50 to-blue-100">
      <div className="h-full max-w-4xl mx-auto flex flex-col p-3">
        <header className="text-center py-4">
          <h1 className="text-4xl font-bold text-blue-500 hover:text-blue-600 transition-colors">
            Bone Fracture AI Chatbot
          </h1>
        </header>

        <div
          ref={chatContainerRef}
          className="flex-1 overflow-y-auto mb-4 rounded-lg bg-white shadow-lg p-4 hide-scrollbar"
        >
          {chatHistory.map((chat, index) => (
            <div
              key={index}
              className={`mb-4 ${
                chat.type === "question" ? "text-right" : "text-left"
              }`}
            >
              <div
                className={`inline-block max-w-[80%] p-3 rounded-lg overflow-auto hide-scrollbar ${
                  chat.type === "question"
                    ? "bg-blue-500 text-white rounded-br-none"
                    : "bg-gray-100 text-gray-800 rounded-bl-none"
                }`}
              >
                <ReactMarkdown className="overflow-auto hide-scrollbar">
                  {chat.content}
                </ReactMarkdown>
              </div>
            </div>
          ))}
        </div>

        <form onSubmit={generateAnswer}>
          <div className="flex items-center space-x-3">
            <input
              type="file"
              accept="image/*"
              onChange={handleFileChange}
              className="flex-1 p-2 border border-gray-300 rounded-lg shadow-md"
            />
            <button
              type="submit"
              className="bg-blue-500 text-white p-2 rounded-lg"
              disabled={generatingAnswer}
            >
              {generatingAnswer ? "Generating..." : "Upload Image"}
            </button>
          </div>
        </form>

        <form onSubmit={handleFollowUpResponse} className="mt-4">
          <div className="flex items-center space-x-3">
            <input
              type="text"
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              placeholder="Type 'yes' for more info or 'no' to skip"
              className="flex-1 p-2 border border-gray-300 rounded-lg shadow-md"
            />
            <button
              type="submit"
              className="bg-blue-500 text-white p-2 rounded-lg"
            >
              Send
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}

export default App;
