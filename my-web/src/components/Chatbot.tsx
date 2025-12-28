import React, { useState, useEffect, useRef } from 'react';
import '../css/Chatbot.css';

interface Message {
  id: number;
  sender: 'user' | 'bot';
  text: string;
}

const Chatbot = () => {
  const [messages, setMessages] = useState<Message[]>([
    { id: 1, sender: 'bot', text: 'Hello! How can I help you today?' }
  ]);
  const [input, setInput] = useState('');
  const [isOpen, setIsOpen] = useState(false);
  const messagesEndRef = useRef<null | HTMLDivElement>(null);

  // Auto-scroll to bottom when messages change
  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const handleSend = () => {
    if (!input.trim()) return;

    const newMessage: Message = {
      id: Date.now(),
      sender: 'user',
      text: input
    };

    setMessages(prev => [...prev, newMessage]);
    setInput('');

    // Placeholder for bot response (for now just echo)
    setTimeout(() => {
      const botResponse: Message = {
        id: Date.now() + 1,
        sender: 'bot',
        text: `I received: "${input}". This is a demo response.`
      };
      setMessages(prev => [...prev, botResponse]);
    }, 500);
  };

  const handleKeyPress = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const toggleChat = () => {
    setIsOpen(!isOpen);
  };

  return (
    <>
      {isOpen ? (
        <div className="chatbot-container">
          <div className="chatbot-header">
            <span>AI Assistant</span>
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
        <button className="chatbot-float-button" onClick={toggleChat}>
          💬
        </button>
      )}
    </>
  );
};

export default Chatbot;
