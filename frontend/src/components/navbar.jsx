import { useState } from "react";
import { Menu, X } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

export default function Navbar() {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <nav className="fixed top-4 left-1/2 transform -translate-x-1/2 w-[90%] md:w-[80%] bg-black border border-white/20 rounded-2xl ">
      <div className="flex justify-between items-center p-4">
    
        <div className="md:hidden">
          <AnimatePresence mode="wait">
            {!isOpen ? (
              <motion.h1
                key="logo"
                initial={{ opacity: 0, y: -10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: 10 }}
                className="text-xl font-bold text-white tracking-wider"
              >
                DNA Disease Analysis
              </motion.h1>
            ) : (
              <motion.ul
                key="menu"
                initial={{ opacity: 0, y: -10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: 10 }}
                className="flex gap-6 text-white"
              >
                {["Home", "About", "Services", "Contact"].map((item) => (
                  <li key={item} className="cursor-pointer hover:text-yellow-300 transition duration-300">
                    {item}
                  </li>
                ))}
              </motion.ul>
            )}
          </AnimatePresence>
        </div>

        {/* Menu Button - Mobile Only */}
        <button onClick={() => setIsOpen(!isOpen)} className="text-white md:hidden">
          {isOpen ? <X size={28} /> : <Menu size={28} />}
        </button>
        <AnimatePresence mode="wait" ><motion.h1 className="hidden md:flex font-bold text-2xl text-white">DNA Disease Analysis</motion.h1></AnimatePresence>
        {/* Desktop Menu (Always Visible) */}
        <ul className="hidden md:flex gap-6 text-white">
          {["Home", "About", "Services", "Contact"].map((item) => (
            <li key={item} className="cursor-pointer hover:text-yellow-300 transition duration-300">
              {item}
            </li>
          ))}
        </ul>
      </div>
    </nav>
  );
}
