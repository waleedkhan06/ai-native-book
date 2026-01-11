"use client"

import type React from "react"
import { useState, useEffect, useRef } from "react"
import "../css/Chatbot.css"

interface Message {
  id: number
  sender: "user" | "bot"
  text: string
}

const API_URL = process.env.NEXT_PUBLIC_API_URL

const Chatbot = () => {
  const [messages, setMessages] = useState<Message[]>([
    { id: 1, sender: "bot", text: "Hello! How can I help you today?" },
  ])
  const [input, setInput] = useState("")
  const [isOpen, setIsOpen] = useState(false)
  const [isTyping, setIsTyping] = useState(false)
  const messagesEndRef = useRef<HTMLDivElement | null>(null)

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [messages, isTyping])

  const getSelectedText = (): string | null => {
    const selection = window.getSelection()
    return selection && selection.toString().trim() ? selection.toString().trim() : null
  }

  const handleSend = async () => {
    if (!input.trim()) return

    if (!API_URL) {
      console.error("NEXT_PUBLIC_API_URL is not defined")
      setMessages((prev) => [
        ...prev,
        {
          id: Date.now(),
          sender: "bot",
          text: "Configuration error: API URL not set.",
        },
      ])
      return
    }

    const userMessage: Message = {
      id: Date.now(),
      sender: "user",
      text: input,
    }

    setMessages((prev) => [...prev, userMessage])
    setInput("")
    setIsTyping(true)

    try {
      const selectedText = getSelectedText()

      const response = await fetch(`${API_URL}/v1/chat/completions`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          question: input,
          selected_text: selectedText,
        }),
      })

      if (!response.ok) {
        throw new Error(`Backend returned ${response.status}`)
      }

      const data = await response.json()

      const botMessage: Message = {
        id: Date.now() + 1,
        sender: "bot",
        text: data.answer || "Sorry, I could not process your request.",
      }

      setMessages((prev) => [...prev, botMessage])
    } catch (err) {
      console.error("Chatbot API error:", err)
      setMessages((prev) => [
        ...prev,
        {
          id: Date.now() + 1,
          sender: "bot",
          text: "Sorry, I could not connect to the server. Please try again.",
        },
      ])
    } finally {
      setIsTyping(false)
    }
  }

  const handleKeyPress = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  return (
    <>
      {isOpen ? (
        <div className="chatbot-container">
          <div className="chatbot-header">
            <span>EmbodiXAI Assistant</span>
            <button className="chatbot-close" onClick={() => setIsOpen(false)}>
              ×
            </button>
          </div>

          <div className="chatbot-messages">
            {messages.map((msg) => (
              <div key={msg.id} className={`chatbot-message ${msg.sender}`}>
                <div className="message-content">{msg.text}</div>
              </div>
            ))}

            {isTyping && (
              <div className="chatbot-message bot">
                <div className="typing-indicator">
                  <div className="typing-dots">
                    <div className="typing-dot" />
                    <div className="typing-dot" />
                    <div className="typing-dot" />
                  </div>
                </div>
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>

          <div className="chatbot-input-container">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Type a message..."
              className="chatbot-input"
            />
            <button onClick={handleSend} className="chatbot-send-button">
              Send
            </button>
          </div>
        </div>
      ) : (
        <button className="chatbot-float-button" onClick={() => setIsOpen(true)}>
          <img src="/img/logo.png" alt="EmbodiXAI Assistant" className="chatbot-logo" />
        </button>
      )}
    </>
  )
}

export default Chatbot
