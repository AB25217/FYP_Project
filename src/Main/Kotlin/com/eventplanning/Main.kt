package com.eventplanning

import com.eventplanning.domain.EventManager
import com.eventplanning.persistance.DataStore
import com.eventplanning.ui.MainWindow
import javax.swing.SwingUtilities
import javax.swing.UIManager

fun main() {

    /**
     * Set global UIManager properties for consistent component styling.
     * these properties make rounded corners for buttons, progress bars, and text components.
     */
    try {
        UIManager.put("Button.arc", 12)
        UIManager.put("Component.arc", 12)
        UIManager.put("ProgressBar.arc", 12)
        UIManager.put("TextComponent.arc", 12)
    } catch (e: Exception) {
        System.err.println("Failed to set UIManager properties: ${e.message}")
    }

    /**
     * creates persistence layer
     * datastore implements repository interface
     * opens a connection to the data store
     * ensures necessary tables are created
     */
    val repository = DataStore()
    repository.connect()
    repository.initializeStorage()

    /**
     * creates the event manager with the repository
     * uses repository to load venues, participants, and events (load and save data)
     */
    val eventManager = EventManager(repository)

    /**
     * loads data into EventManager
     */
    if (eventManager.initializeData()) {
        println("Data loaded successfully.")
    } else {
        System.err.println("Warning: Failed to load some data.")
    }

    /**
     * Launches the main application window
     * swing requires UI updates to be done on the Event Dispatch Thread (EDT)
     * create main window and give it EventManager
     * therefore gui can display and manipulate event data
     */
    SwingUtilities.invokeLater {
        val mainWindow = MainWindow(eventManager)
        mainWindow.show()
    }
    /**
     * Registers a shutdown hook
     * save data and disconnect from the repository when the application exits.
     */
    Runtime.getRuntime().addShutdownHook(Thread {
        println("Shutting down...")
        eventManager.saveAllData()
        repository.disconnect()
    })
}