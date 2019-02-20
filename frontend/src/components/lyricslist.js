import React from 'react';

export default class LyricsList extends React.Component {
    renderLyrics() {
        const lyrics = Object.values((this.props.lyrics))

        return lyrics.map(lyric => <div><h2>{lyric.title}</h2></div> )
    }

    render() {
        return (
            <div>
                {this.renderLyrics()}
            </div>
        )
    }

}