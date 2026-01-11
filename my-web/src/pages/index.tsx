import Link from "@docusaurus/Link";
import Layout from "@theme/Layout";
import { FaBook, FaRobot } from "react-icons/fa";
import CardComponent from "@site/src/components/CardComponent";

export default function Home(): JSX.Element {
  return (
    <Layout title="Physical AI & Humanoid Robotics Textbook" description="The Complete Guide to Embodied Intelligence">
      <div className="floating-dots-fullpage">
        {/* Hero Section */}
        <header className="text--center padding-vert--xl">
          <div className="container">
            <h1 className="main-title">
              Physical AI & Humanoid <br />
              Robotics Textbook
            </h1>
            <p className="subtitle">
              The Complete Guide to Embodied Intelligence. Master the future of robotics through interactive AI-powered
              learning and hands-on projects.
            </p>
            {/* ✅ FIX: Add .hero__button-group wrapper */}
            <div className="hero__button-group margin-top--lg">
              <Link
                to="/docs/module-1/"
                className="button button--secondary button--lg"
                style={{ backgroundColor: "#9333EA", border: "none", color: "white" }}
              >
                <FaBook style={{ marginRight: "10px" }} />
                Explore Textbook
              </Link>
              <Link
                to="/docs/tutorial"
                className="button button--secondary button--lg"
                style={{ backgroundColor: "#A855F7", border: "none", color: "white" }}
              >
                <FaRobot style={{ marginRight: "10px" }} />
                AI Book Tutorial
              </Link>
            </div>
          </div>
        </header>

        {/* Cards + Content */}
        <main className="container padding-bottom--xl">
          <CardComponent />

          <div className="margin-top--xl text--left doc-content" style={{ maxWidth: "800px", margin: "4rem auto 0" }}>
            <h2>Overview</h2>
            <p>
              Welcome to the Physical AI & Humanoid Robotics textbook. This comprehensive resource is designed to take
              you from the fundamentals of robotics to advanced topics in embodied intelligence. Whether you're a
              student, researcher, or hobbyist, you'll find everything you need to build and deploy intelligent robotic
              systems.
            </p>

            <div className="margin-top--lg">
              <h3>Learning Objectives</h3>
              <ul>
                <li>Understand the fundamental concepts of Physical AI and embodied intelligence.</li>
                <li>Master ROS2 architecture and middleware for complex robotic systems.</li>
                <li>Build and simulate humanoid robots in physics-based environments like Gazebo and Unity.</li>
                <li>Implement advanced perception using NVIDIA Isaac and computer vision.</li>
                <li>Explore the intersection of Vision, Language, and Action in modern robotics.</li>
              </ul>
            </div>
          </div>
        </main>
      </div>
    </Layout>
  );
} 