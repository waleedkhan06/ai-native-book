import React from 'react';

interface ParticleBackgroundProps {
  className?: string;
}

const ParticleBackground: React.FC<ParticleBackgroundProps> = ({ className = '' }) => {
  // Create 50+ animated circles with varying properties
  const particles = Array.from({ length: 60 }, (_, i) => {
    const size = Math.random() * 10 + 2; // Random size between 2px and 12px
    const posX = Math.random() * 100; // Random position X percentage
    const posY = Math.random() * 100; // Random position Y percentage
    const animationDuration = Math.random() * 20 + 10; // Random animation duration between 10-30s
    const animationDelay = Math.random() * 5; // Random delay between 0-5s
    const opacity = Math.random() * 0.5 + 0.1; // Random opacity between 0.1-0.6
    const isPurple = Math.random() > 0.5; // Randomly choose purple or white

    return {
      id: i,
      size: `${size}px`,
      top: `${posY}%`,
      left: `${posX}%`,
      animationDuration: `${animationDuration}s`,
      animationDelay: `${animationDelay}s`,
      opacity,
      backgroundColor: isPurple ? '#8B5CF6' : '#FFFFFF',
    };
  });

  return (
    <div className={`particle-background ${className}`}>
      {particles.map((particle) => (
        <div
          key={particle.id}
          className="particle"
          style={{
            position: 'absolute',
            width: particle.size,
            height: particle.size,
            top: particle.top,
            left: particle.left,
            backgroundColor: particle.backgroundColor,
            borderRadius: '50%',
            opacity: particle.opacity,
            animation: `float ${particle.animationDuration} ease-in-out infinite`,
            animationDelay: particle.animationDelay,
            willChange: 'transform',
          }}
        />
      ))}
      <style jsx>{`
        .particle-background {
          position: fixed;
          top: 0;
          left: 0;
          width: 100%;
          height: 100%;
          pointer-events: none;
          z-index: -1;
          overflow: hidden;
        }

        @keyframes float {
          0% {
            transform: translate(0, 0) rotate(0deg);
          }
          25% {
            transform: translate(${Math.random() * 100 - 50}px, ${Math.random() * 100 - 50}px) rotate(90deg);
          }
          50% {
            transform: translate(${Math.random() * 100 - 50}px, ${Math.random() * 100 - 50}px) rotate(180deg);
          }
          75% {
            transform: translate(${Math.random() * 100 - 50}px, ${Math.random() * 100 - 50}px) rotate(270deg);
          }
          100% {
            transform: translate(0, 0) rotate(360deg);
          }
        }
      `}</style>
    </div>
  );
};

export default ParticleBackground;